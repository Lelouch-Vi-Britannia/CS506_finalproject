import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import optuna
import optuna.visualization as vis


# 定义数据预处理函数
def preprocess_data(input_file, sequence_length, scaler=None, fit_scaler=False, forecast_horizon=24):
    """
    预处理数据，创建序列和目标变量。

    参数：
    - input_file: 输入的CSV文件路径。
    - sequence_length: 输入序列的长度。
    - scaler: 训练时拟合的StandardScaler对象（用于测试集）。
    - fit_scaler: 是否在此数据上拟合scaler（仅训练集）。
    - forecast_horizon: 预测的时间步数（默认24小时）。

    返回：
    - X, y: 特征和目标变量的numpy数组。
    - scaler: 拟合的StandardScaler对象（仅训练集）。
    """
    data = pd.read_csv(input_file)
    # 选择特征和目标
    features = ['Temperature', 'Relative Humidity', 'Dwpt', 'Date', 'Month', 'Time (est)']
    target = 'Temperature'  # 预测最高温度

    # 删除缺失值
    data = data.dropna(subset=features + [target])

    # 特征工程
    # 提取小时信息
    data['Hour'] = data['Time (est)'].apply(lambda x: int(x.split(':')[0]))

    # 将日期转换为周期性特征
    data['Date_Sin'] = np.sin(2 * np.pi * data['Date'] / 31)
    data['Date_Cos'] = np.cos(2 * np.pi * data['Date'] / 31)

    # 将月份转换为周期性特征
    data['Month_Sin'] = np.sin(2 * np.pi * data['Month'] / 12)
    data['Month_Cos'] = np.cos(2 * np.pi * data['Month'] / 12)

    # 将小时转换为周期性特征
    data['Hour_Sin'] = np.sin(2 * np.pi * data['Hour'] / 24)
    data['Hour_Cos'] = np.cos(2 * np.pi * data['Hour'] / 24)

    # 创建过去24小时的温度滞后特征
    for lag in range(1, 25):
        data[f'Temperature_lag_{lag}'] = data['Temperature'].shift(lag)

    # 删除滞后特征中存在缺失值的行
    data = data.dropna(subset=[f'Temperature_lag_{lag}' for lag in range(1, 25)])

    # 选择最终特征
    final_features = ['Temperature', 'Relative Humidity', 'Dwpt',
                      'Date_Sin', 'Date_Cos', 'Month_Sin', 'Month_Cos',
                      'Hour_Sin', 'Hour_Cos'] + [f'Temperature_lag_{lag}' for lag in range(1, 25)]

    # 标准化特征
    if scaler is None:
        scaler = StandardScaler()
    if fit_scaler:
        data[final_features] = scaler.fit_transform(data[final_features])
    else:
        data[final_features] = scaler.transform(data[final_features])

    # 创建目标变量：未来24小时的最高温度
    data['Max_Temp_24h'] = data['Temperature'].rolling(window=forecast_horizon).max().shift(-forecast_horizon + 1)

    # 删除无法计算目标变量的行
    data = data.dropna(subset=['Max_Temp_24h'])

    # 创建输入和目标序列
    X = data[final_features].values
    y = data['Max_Temp_24h'].values

    # 创建序列数据
    def create_sequences(data, seq_length):
        X_seq = []
        y_seq = []
        for i in range(len(data) - seq_length):
            X_seq.append(data[i:i + seq_length, :-1])  # 特征
            y_seq.append(data[i + seq_length, -1])  # 目标最高温度
        return np.array(X_seq), np.array(y_seq)

    X, y = create_sequences(np.column_stack((X, y)), sequence_length)

    return X, y, scaler


# 定义模型
def define_model(input_size, hidden_size, num_layers, output_size, dropout=0.0):
    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
            super(LSTMModel, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers

            # 定义LSTM层
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

            # 定义全连接层
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            # 初始化隐藏状态和细胞状态
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

            # 前向传播LSTM
            out, _ = self.lstm(x, (h0, c0))  # out: (batch, seq_length, hidden_size)

            # 只取最后一个时间步的输出
            out = out[:, -1, :]  # (batch, hidden_size)

            # 全连接层
            out = self.fc(out)  # (batch, output_size)
            return out

    model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout)
    return model


# 定义训练和评估函数
def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=100, patience=10):
    best_val_loss = float('inf')
    trigger_times = 0

    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        start_time = time.time()

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # 前向传播
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        # 计算训练集的平均损失
        avg_train_loss = np.mean(train_losses)

        # 评估验证集
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch = X_val_batch.to(device)
                y_val_batch = y_val_batch.to(device)

                val_outputs = model(X_val_batch)
                val_loss = criterion(val_outputs.squeeze(), y_val_batch)
                val_losses.append(val_loss.item())

        avg_val_loss = np.mean(val_losses)

        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)

        # print(f'Epoch [{epoch + 1}/{num_epochs}], '
        #       f'Train Loss: {avg_train_loss:.4f}, '
        #       f'Val Loss: {avg_val_loss:.4f}, '
        #       f'Time: {int(epoch_mins)}m {int(epoch_secs)}s')

        # 早停法
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            trigger_times = 0
            torch.save(model.state_dict(), 'best_lstm_model.pth')
            # print("保存最优模型。")
        else:
            trigger_times += 1
            # print(f'早停次数: {trigger_times}')
            if trigger_times >= patience:
                # print('早停法触发，停止训练。')
                break

    return best_val_loss


# 定义Dataset类
class TemperatureDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# 定义Optuna的目标函数
def objective(trial):
    # 超参数采样
    sequence_length = trial.suggest_int('sequence_length', 12, 48)  # 12到48小时
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    hidden_size = trial.suggest_int('hidden_size', 16, 128)
    num_layers = trial.suggest_int('num_layers', 1, 4)
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)

    # 数据预处理
    X, y, scaler = preprocess_data('data/1.csv', sequence_length, fit_scaler=True, forecast_horizon=24)

    # 拆分数据集
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False, random_state=42)

    # 创建数据加载器
    train_dataset = TemperatureDataset(X_train, y_train)
    val_dataset = TemperatureDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 定义模型
    input_size = X_train.shape[2]
    output_size = 1
    model = define_model(input_size, hidden_size, num_layers, output_size, dropout=0.0).to(device)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 训练模型
    best_val_loss = train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, device,
                                       num_epochs=100, patience=10)

    return best_val_loss


# 主函数
if __name__ == "__main__":
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    sequence_length = 14
    batch_size = 32
    # 预处理训练集和验证集
    X, y, scaler = preprocess_data('data/1.csv', sequence_length, fit_scaler=True, forecast_horizon=24)

    # 拆分数据集
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False, random_state=42)

    # 创建数据加载器
    train_dataset = TemperatureDataset(X_train, y_train)
    val_dataset = TemperatureDataset(X_val, y_val)
    test_dataset = TemperatureDataset(X_test, y_test)  # 原始测试集
    external_test_dataset = TemperatureDataset(
        *preprocess_data('data/test.csv', sequence_length, scaler=scaler, fit_scaler=False, forecast_horizon=24)[
         :2])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # 原始测试集
    external_test_loader = DataLoader(external_test_dataset, batch_size=batch_size, shuffle=False)  # 外部测试集

    # 定义模型
    input_size = X_train.shape[2]
    output_size = 1
    # model = define_model(input_size, best_params['hidden_size'], best_params['num_layers'], output_size,
    #                      dropout=0.0).to(device)
    model = define_model(input_size, 121, 3, output_size,
                         dropout=0.0).to(device)
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002930386152314192)
    # 训练模型（使用较大的epoch和耐心）
    # print("开始训练最佳模型...")
    best_val_loss = train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, device,
                                       num_epochs=500, patience=20)

    # 加载最优模型
    model.load_state_dict(torch.load('best_lstm_model.pth'))
    model.to(device)
    model.eval()

    # 在外部测试集上进行预测
    external_test_losses = []
    external_all_predictions = []
    external_all_targets = []

    with torch.no_grad():
        for X_ext_batch, y_ext_batch in val_loader:
            X_ext_batch = X_ext_batch.to(device)
            y_ext_batch = y_ext_batch.to(device)

            ext_outputs = model(X_ext_batch)
            ext_loss = criterion(ext_outputs.squeeze(), y_ext_batch)
            external_test_losses.append(ext_loss.item())

            external_all_predictions.append(ext_outputs.cpu().numpy())
            external_all_targets.append(y_ext_batch.cpu().numpy())

    avg_external_test_loss = np.mean(external_test_losses)
    # print(f'外部测试集平均损失: {avg_external_test_loss:.4f}')

    # 反标准化目标和预测
    y_test_external = np.concatenate(external_all_targets, axis=0)
    y_pred_external = np.concatenate(external_all_predictions, axis=0).squeeze()

    y_test_external_actual = y_test_external * scaler.scale_[0] + scaler.mean_[0]
    y_pred_external_actual = y_pred_external * scaler.scale_[0] + scaler.mean_[0]

    # 计算评估指标，如RMSE和MAE
    rmse_external = np.sqrt(mean_squared_error(y_test_external_actual, y_pred_external_actual))
    mae_external = mean_absolute_error(y_test_external_actual, y_pred_external_actual)
    print("with lr:" + str(0.0002930386152314192))
    print(f'内部验证集 RMSE: {rmse_external:.2f} ºF')
    print(f'内部验证集 MAE: {mae_external:.2f} ºF')
    external_test_losses = []
    external_all_predictions = []
    external_all_targets = []

    with torch.no_grad():
        for X_ext_batch, y_ext_batch in external_test_loader:
            X_ext_batch = X_ext_batch.to(device)
            y_ext_batch = y_ext_batch.to(device)

            ext_outputs = model(X_ext_batch)
            ext_loss = criterion(ext_outputs.squeeze(), y_ext_batch)
            external_test_losses.append(ext_loss.item())

            external_all_predictions.append(ext_outputs.cpu().numpy())
            external_all_targets.append(y_ext_batch.cpu().numpy())

    avg_external_test_loss = np.mean(external_test_losses)
    # print(f'外部测试集平均损失: {avg_external_test_loss:.4f}')

    # 反标准化目标和预测
    y_test_external = np.concatenate(external_all_targets, axis=0)
    y_pred_external = np.concatenate(external_all_predictions, axis=0).squeeze()

    y_test_external_actual = y_test_external * scaler.scale_[0] + scaler.mean_[0]
    y_pred_external_actual = y_pred_external * scaler.scale_[0] + scaler.mean_[0]

    # 计算评估指标，如RMSE和MAE
    rmse_external = np.sqrt(mean_squared_error(y_test_external_actual, y_pred_external_actual))
    mae_external = mean_absolute_error(y_test_external_actual, y_pred_external_actual)
    print(f'外部测试集 RMSE: {rmse_external:.2f} ºF')
    print(f'外部测试集 MAE: {mae_external:.2f} ºF')

    # 可视化预测结果
    # 确保y_test_external_actual和y_pred_external_actual的长度相同
    assert len(y_test_external_actual) == len(y_pred_external_actual), "实际值和预测值长度不一致！"

    # 创建样本索引
    external_sample_indices = range(len(y_test_external_actual))

    plt.figure(figsize=(15, 7))
    plt.scatter(external_sample_indices, y_test_external_actual, label='实际最高温度', alpha=0.5, s=10)
    plt.scatter(external_sample_indices, y_pred_external_actual, label='预测最高温度', alpha=0.5, s=10)
    plt.xlabel('样本索引')
    plt.ylabel('最高温度 (ºF)')
    plt.title('预测与实际最高温度对比（外部测试集）')
    plt.legend()
    plt.show()

    # 可选：绘制实际值与预测值的关系图
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test_external_actual, y_pred_external_actual, alpha=0.5, s=10)
    plt.plot([y_test_external_actual.min(), y_test_external_actual.max()],
             [y_test_external_actual.min(), y_test_external_actual.max()], 'r--')  # y=x线
    plt.xlabel('实际最高温度 (ºF)')
    plt.ylabel('预测最高温度 (ºF)')
    plt.title('实际值与预测值关系图（外部测试集）')
    plt.show()

    # 结束
    print("模型训练和评估完成。")
