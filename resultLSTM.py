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


# Define data preprocessing function
def preprocess_data(input_file, sequence_length, scaler=None, fit_scaler=False, forecast_horizon=24):
    """
    Preprocess data, create sequences and target variables.

    Parameters:
    - input_file: Path to the input CSV file.
    - sequence_length: Length of the input sequence.
    - scaler: A fitted StandardScaler object (for test sets).
    - fit_scaler: Whether to fit the scaler on this dataset (training sets only).
    - forecast_horizon: Forecast horizon (default 24 hours).

    Returns:
    - X, y: Numpy arrays of features and target variables.
    - scaler: The fitted StandardScaler object (training sets only).
    """
    data = pd.read_csv(input_file)
    # Select features and target
    features = ['Temperature', 'Relative Humidity', 'Dwpt', 'Date', 'Month', 'Time (est)']
    target = 'Temperature'  # Predicting maximum temperature

    # Drop missing values
    data = data.dropna(subset=features + [target])

    # Feature engineering
    # Extract hour information
    data['Hour'] = data['Time (est)'].apply(lambda x: int(x.split(':')[0]))

    # Convert date to cyclical features
    data['Date_Sin'] = np.sin(2 * np.pi * data['Date'] / 31)
    data['Date_Cos'] = np.cos(2 * np.pi * data['Date'] / 31)

    # Convert month to cyclical features
    data['Month_Sin'] = np.sin(2 * np.pi * data['Month'] / 12)
    data['Month_Cos'] = np.cos(2 * np.pi * data['Month'] / 12)

    # Convert hour to cyclical features
    data['Hour_Sin'] = np.sin(2 * np.pi * data['Hour'] / 24)
    data['Hour_Cos'] = np.cos(2 * np.pi * data['Hour'] / 24)

    # Create past 24-hour temperature lag features
    for lag in range(1, 25):
        data[f'Temperature_lag_{lag}'] = data['Temperature'].shift(lag)

    # Drop rows with missing values in lag features
    data = data.dropna(subset=[f'Temperature_lag_{lag}' for lag in range(1, 25)])

    # Final feature selection
    final_features = ['Temperature', 'Relative Humidity', 'Dwpt',
                      'Date_Sin', 'Date_Cos', 'Month_Sin', 'Month_Cos',
                      'Hour_Sin', 'Hour_Cos'] + [f'Temperature_lag_{lag}' for lag in range(1, 25)]

    # Standardize features
    if scaler is None:
        scaler = StandardScaler()
    if fit_scaler:
        data[final_features] = scaler.fit_transform(data[final_features])
    else:
        data[final_features] = scaler.transform(data[final_features])

    # Create target variable: maximum temperature in the next 24 hours
    data['Max_Temp_24h'] = data['Temperature'].rolling(window=forecast_horizon).max().shift(-forecast_horizon + 1)

    # Drop rows where target cannot be computed
    data = data.dropna(subset=['Max_Temp_24h'])

    # Create input and target sequences
    X = data[final_features].values
    y = data['Max_Temp_24h'].values

    # Create sequence data
    def create_sequences(data, seq_length):
        X_seq = []
        y_seq = []
        for i in range(len(data) - seq_length):
            X_seq.append(data[i:i + seq_length, :-1])  # Features
            y_seq.append(data[i + seq_length, -1])  # Target max temperature
        return np.array(X_seq), np.array(y_seq)

    X, y = create_sequences(np.column_stack((X, y)), sequence_length)

    return X, y, scaler


# Define the model
def define_model(input_size, hidden_size, num_layers, output_size, dropout=0.0):
    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
            super(LSTMModel, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers

            # Define LSTM layer
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

            # Define fully connected layer
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            # Initialize hidden and cell states
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

            # Forward LSTM
            out, _ = self.lstm(x, (h0, c0))  # out: (batch, seq_length, hidden_size)

            # Take the output of the last time step
            out = out[:, -1, :]  # (batch, hidden_size)

            # Fully connected layer
            out = self.fc(out)  # (batch, output_size)
            return out

    model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout)
    return model


# Define training and evaluation functions
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

            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        # Compute average training loss
        avg_train_loss = np.mean(train_losses)

        # Evaluate on validation set
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

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            trigger_times = 0
            torch.save(model.state_dict(), 'best_lstm_model.pth')
        else:
            trigger_times += 1
            if trigger_times >= patience:
                break

    return best_val_loss


# Define Dataset class
class TemperatureDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Define Optuna objective function
def objective(trial):
    # Hyperparameter sampling
    sequence_length = trial.suggest_int('sequence_length', 12, 48)  # 12 to 48 hours
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    hidden_size = trial.suggest_int('hidden_size', 16, 128)
    num_layers = trial.suggest_int('num_layers', 1, 4)
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)

    # Data preprocessing
    X, y, scaler = preprocess_data('data/1.csv', sequence_length, fit_scaler=True, forecast_horizon=24)

    # Split dataset
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False, random_state=42)

    # Create data loaders
    train_dataset = TemperatureDataset(X_train, y_train)
    val_dataset = TemperatureDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Define model
    input_size = X_train.shape[2]
    output_size = 1
    model = define_model(input_size, hidden_size, num_layers, output_size, dropout=0.0).to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train model
    best_val_loss = train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, device,
                                       num_epochs=100, patience=10)

    return best_val_loss


# Main function
if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    sequence_length = 14
    batch_size = 32
    # Preprocess training and validation sets
    X, y, scaler = preprocess_data('data/1.csv', sequence_length, fit_scaler=True, forecast_horizon=24)

    # Split dataset
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False, random_state=42)

    # Create data loaders
    train_dataset = TemperatureDataset(X_train, y_train)
    val_dataset = TemperatureDataset(X_val, y_val)
    test_dataset = TemperatureDataset(X_test, y_test)  # Original test set
    external_test_dataset = TemperatureDataset(
        *preprocess_data('data/test.csv', sequence_length, scaler=scaler, fit_scaler=False, forecast_horizon=24)[
         :2])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # Original test set
    external_test_loader = DataLoader(external_test_dataset, batch_size=batch_size, shuffle=False)  # External test set

    # Define model
    input_size = X_train.shape[2]
    output_size = 1
    model = define_model(input_size, 121, 3, output_size,
                         dropout=0.0).to(device)
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002930386152314192)
    # Train model (with larger epochs and patience)
    best_val_loss = train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, device,
                                       num_epochs=500, patience=20)

    # Load the best model
    model.load_state_dict(torch.load('best_lstm_model.pth'))
    model.to(device)
    model.eval()

    # Predict on external test set (internal validation set)
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

    # Inverse transform target and predictions
    y_test_external = np.concatenate(external_all_targets, axis=0)
    y_pred_external = np.concatenate(external_all_predictions, axis=0).squeeze()

    y_test_external_actual = y_test_external * scaler.scale_[0] + scaler.mean_[0]
    y_pred_external_actual = y_pred_external * scaler.scale_[0] + scaler.mean_[0]

    # Compute evaluation metrics such as RMSE and MAE
    rmse_external = np.sqrt(mean_squared_error(y_test_external_actual, y_pred_external_actual))
    mae_external = mean_absolute_error(y_test_external_actual, y_pred_external_actual)
    print("with lr:" + str(0.0002930386152314192))
    print(f'Internal validation set RMSE: {rmse_external:.2f} ºF')
    print(f'Internal validation set MAE: {mae_external:.2f} ºF')

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

    # Inverse transform target and predictions
    y_test_external = np.concatenate(external_all_targets, axis=0)
    y_pred_external = np.concatenate(external_all_predictions, axis=0).squeeze()

    y_test_external_actual = y_test_external * scaler.scale_[0] + scaler.mean_[0]
    y_pred_external_actual = y_pred_external * scaler.scale_[0] + scaler.mean_[0]

    # Compute evaluation metrics such as RMSE and MAE
    rmse_external = np.sqrt(mean_squared_error(y_test_external_actual, y_pred_external_actual))
    mae_external = mean_absolute_error(y_test_external_actual, y_pred_external_actual)
    print(f'External test set RMSE: {rmse_external:.2f} ºF')
    print(f'External test set MAE: {mae_external:.2f} ºF')

    # Visualization of the predictions
    # Ensure y_test_external_actual and y_pred_external_actual have the same length
    assert len(y_test_external_actual) == len(y_pred_external_actual), "Mismatch in length of actual and predicted values!"

    # Create sample indices
    external_sample_indices = range(len(y_test_external_actual))

    plt.figure(figsize=(15, 7))
    plt.scatter(external_sample_indices, y_test_external_actual, label='Actual Max Temperature', alpha=0.5, s=10)
    plt.scatter(external_sample_indices, y_pred_external_actual, label='Predicted Max Temperature', alpha=0.5, s=10)
    plt.xlabel('Sample Index')
    plt.ylabel('Max Temperature (ºF)')
    plt.title('Comparison of Predicted vs Actual Max Temperature (External Test Set)')
    plt.legend()
    plt.show()

    # Optional: plot actual vs predicted
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test_external_actual, y_pred_external_actual, alpha=0.5, s=10)
    plt.plot([y_test_external_actual.min(), y_test_external_actual.max()],
             [y_test_external_actual.min(), y_test_external_actual.max()], 'r--')  # y=x line
    plt.xlabel('Actual Max Temperature (ºF)')
    plt.ylabel('Predicted Max Temperature (ºF)')
    plt.title('Actual vs Predicted (External Test Set)')
    plt.show()

    # Done
    print("Model training and evaluation completed.")
