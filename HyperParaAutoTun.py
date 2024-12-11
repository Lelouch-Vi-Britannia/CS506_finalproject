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
def preprocess_data(input_file, sequence_length, forecast_horizon=24):
    data = pd.read_csv(input_file)

    # Select features and target
    features = ['Temperature', 'Relative Humidity', 'Dwpt', 'Date', 'Month', 'Time (est)']
    target = 'Temperature'  # Predicting max temperature

    # Drop missing values
    data = data.dropna(subset=features + [target])

    # Feature engineering
    # Extract hour information
    data['Hour'] = data['Time (est)'].apply(lambda x: int(x.split(':')[0]))

    # Convert date into cyclical features
    data['Date_Sin'] = np.sin(2 * np.pi * data['Date'] / 31)
    data['Date_Cos'] = np.cos(2 * np.pi * data['Date'] / 31)

    # Convert month into cyclical features
    data['Month_Sin'] = np.sin(2 * np.pi * data['Month'] / 12)
    data['Month_Cos'] = np.cos(2 * np.pi * data['Month'] / 12)

    # Convert hour into cyclical features
    data['Hour_Sin'] = np.sin(2 * np.pi * data['Hour'] / 24)
    data['Hour_Cos'] = np.cos(2 * np.pi * data['Hour'] / 24)

    # Create lag features for the past 24 hours of temperature
    for lag in range(1, 25):
        data[f'Temperature_lag_{lag}'] = data['Temperature'].shift(lag)

    # Drop rows with missing values in lag features
    data = data.dropna(subset=[f'Temperature_lag_{lag}' for lag in range(1, 25)])

    # Final features
    final_features = ['Temperature', 'Relative Humidity', 'Dwpt',
                      'Date_Sin', 'Date_Cos', 'Month_Sin', 'Month_Cos',
                      'Hour_Sin', 'Hour_Cos'] + [f'Temperature_lag_{lag}' for lag in range(1, 25)]

    # Standardize features
    scaler = StandardScaler()
    data[final_features] = scaler.fit_transform(data[final_features])

    # Create target variable: max temperature in the next 24 hours
    data['Max_Temp_24h'] = data['Temperature'].rolling(window=forecast_horizon).max().shift(-forecast_horizon + 1)

    # Drop rows where the target cannot be computed
    data = data.dropna(subset=['Max_Temp_24h'])

    # Create input and target sequences
    X = data[final_features].values
    y = data['Max_Temp_24h'].values

    # Create sequence data
    def create_sequences(data, seq_length):
        X_seq = []
        y_seq = []
        for i in range(len(data) - seq_length):
            X_seq.append(data[i:i + seq_length, :-1])  # features
            y_seq.append(data[i + seq_length, -1])     # target max temperature
        return np.array(X_seq), np.array(y_seq)

    X, y = create_sequences(np.column_stack((X, y)), sequence_length)

    return X, y, scaler

# Define model
def define_model(input_size, hidden_size, num_layers, output_size, dropout=0.2):
    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
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

            # Take the output at the last time step
            out = out[:, -1, :]  # (batch, hidden_size)

            # Fully connected layer
            out = self.fc(out)  # (batch, output_size)
            return out

    model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout)
    return model

# Define train and evaluation function
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

        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Train Loss: {avg_train_loss:.4f}, '
              f'Val Loss: {avg_val_loss:.4f}, '
              f'Time: {int(epoch_mins)}m {int(epoch_secs)}s')

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            trigger_times = 0
            torch.save(model.state_dict(), 'best_lstm_model.pth')
            print("Saved the best model.")
        else:
            trigger_times += 1
            print(f'Early stopping count: {trigger_times}')
            if trigger_times >= patience:
                print('Early stopping triggered, stopping training.')
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

# Define the Optuna objective function
def objective(trial):
    # Hyperparameter sampling
    sequence_length = trial.suggest_int('sequence_length', 12, 48)  # e.g., 12 to 48 hours
    batch_size = trial.suggest_categorical('batch_size', [16,32, 64, 128])
    hidden_size = trial.suggest_int('hidden_size', 16, 128)
    num_layers = trial.suggest_int('num_layers', 1, 4)
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)

    # Data preprocessing
    X, y, scaler = preprocess_data('data/1.csv', sequence_length)

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
    model = define_model(input_size, hidden_size, num_layers, output_size).to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train model
    best_val_loss = train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, device,
                                      num_epochs=100, patience=15)

    return best_val_loss

# Main function
if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    #
    # Create Optuna study
    study = optuna.create_study(direction='minimize')
    print("Starting hyperparameter optimization...")
    study.optimize(objective, n_trials=200)

    # Print best results
    print("Best validation loss: ", study.best_value)
    print("Best hyperparameters: ", study.best_params)

    # Evaluate model with best hyperparameters
    best_params = study.best_params
    sequence_length = best_params['sequence_length']
    batch_size = best_params['batch_size']
    # sequence_length = 25
    # batch_size = 32
    # Preprocess data
    X, y, scaler = preprocess_data('data/1.csv', sequence_length)

    # Split dataset
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False, random_state=42)

    # Create data loaders
    train_dataset = TemperatureDataset(X_train, y_train)
    val_dataset = TemperatureDataset(X_val, y_val)
    test_dataset = TemperatureDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Define model
    input_size = X_train.shape[2]
    output_size = 1
    model = define_model(input_size, best_params['hidden_size'], best_params['num_layers'], output_size,
                         dropout=0.0).to(device)
    # model = define_model(input_size, 128, 3, output_size,
    #                      dropout=0.0).to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'])
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0004347147277109377)
    # Train model (with more epochs and patience)
    print("Starting to train the best model...")
    best_val_loss = train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, device,
                                      num_epochs=500, patience=50)

    # Load the best model
    model.load_state_dict(torch.load('best_lstm_model.pth'))
    model.to(device)
    model.eval()

    # Predict on test set
    test_losses = []
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for X_test_batch, y_test_batch in test_loader:
            X_test_batch = X_test_batch.to(device)
            y_test_batch = y_test_batch.to(device)

            test_outputs = model(X_test_batch)
            test_loss = criterion(test_outputs.squeeze(), y_test_batch)
            test_losses.append(test_loss.item())

            all_predictions.append(test_outputs.cpu().numpy())
            all_targets.append(y_test_batch.cpu().numpy())

    avg_test_loss = np.mean(test_losses)
    print(f'Test set average loss: {avg_test_loss:.4f}')

    # Inverse transform for target and prediction
    y_test = np.concatenate(all_targets, axis=0)
    y_pred = np.concatenate(all_predictions, axis=0).squeeze()

    y_test_actual = y_test * scaler.scale_[0] + scaler.mean_[0]
    y_pred_actual = y_pred * scaler.scale_[0] + scaler.mean_[0]

    # Evaluate metrics: RMSE and MAE
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
    mae = mean_absolute_error(y_test_actual, y_pred_actual)

    print(f'Test set RMSE: {rmse:.2f} ºF')
    print(f'Test set MAE: {mae:.2f} ºF')

    # Visualization
    # Ensure lengths match
    assert len(y_test_actual) == len(y_pred_actual), "Length mismatch between actual and predicted values!"

    # Create sample indices
    sample_indices = range(len(y_test_actual))

    plt.figure(figsize=(15, 7))
    plt.scatter(sample_indices, y_test_actual, label='Actual Max Temperature', alpha=0.5, s=10)
    plt.scatter(sample_indices, y_pred_actual, label='Predicted Max Temperature', alpha=0.5, s=10)
    plt.xlabel('Sample Index')
    plt.ylabel('Max Temperature (ºF)')
    plt.title('Comparison of Predicted vs Actual Max Temperature')
    plt.legend()
    plt.show()

    # Optional: plot actual vs predicted
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test_actual, y_pred_actual, alpha=0.5, s=10)
    plt.plot([y_test_actual.min(), y_test_actual.max()], [y_test_actual.min(), y_test_actual.max()], 'r--')  # y=x line
    plt.xlabel('Actual Max Temperature (ºF)')
    plt.ylabel('Predicted Max Temperature (ºF)')
    plt.title('Actual vs Predicted Scatter Plot')
    plt.show()

    # Done
    print("Model training and evaluation completed.")
