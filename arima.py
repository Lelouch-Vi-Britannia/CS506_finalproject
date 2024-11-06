import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# Suppress warnings related to non-invertible MA parameters and convergence issues
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="Non-stationary starting autoregressive parameters found.")
warnings.filterwarnings("ignore", message="Non-invertible starting MA parameters found.")

# Load data from 'weather_data.csv'
weather_data = pd.read_csv('./data/weather_data.csv')

# Display the first 5 rows to verify data
print("First 5 rows of the dataset:")
print(weather_data.head())

# Correct the 'Datetime' column by replacing the year with 2024 and month with 10
def correct_datetime(row):
    # Convert 'Datetime' string to datetime object
    date_time = pd.to_datetime(row['Datetime'])
    # Replace year with 2024, month with 10
    corrected_datetime = date_time.replace(year=2024, month=10)
    return corrected_datetime

# Apply correction to 'Datetime' column
weather_data['Datetime'] = weather_data.apply(correct_datetime, axis=1)

# Keep only 'Datetime' and 'Temperature (ºF)' columns
weather_data = weather_data[['Datetime', 'Temperature (ºF)']]

# Rename columns for consistency
weather_data.rename(columns={'Datetime': 'DATE', 'Temperature (ºF)': 'TEMP_F'}, inplace=True)

# Convert 'DATE' column to datetime format
weather_data['DATE'] = pd.to_datetime(weather_data['DATE'])

# Sort data by date in ascending order
weather_data.sort_values(by='DATE', inplace=True)

# Reset index
weather_data.reset_index(drop=True, inplace=True)

# Display cleaned data
print("\nCleaned data:")
print(weather_data.head())

# Use all data except the last 24 hours for training, and the last 24 hours for testing
train_data = weather_data.iloc[:-24]
test_data = weather_data.iloc[-24:]

print(f"\nTraining set size: {len(train_data)}")
print(f"Test set size: {len(test_data)}")

plt.figure(figsize=(12, 6))
plt.plot(train_data['DATE'], train_data['TEMP_F'], marker='o', linestyle='-')
plt.title('Temperature Over Time (Training Data)')
plt.xlabel('Date')
plt.ylabel('Temperature (°F)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


def test_stationarity(timeseries):
    print('Dickey-Fuller Test Results:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput[f'Critical Value ({key})'] = value
    print(dfoutput)


plt.figure(figsize=(12,6))
plt.subplot(121)
plot_acf(train_data['TEMP_F'], ax=plt.gca(), lags=20)
plt.subplot(122)
plot_pacf(train_data['TEMP_F'], ax=plt.gca(), lags=20)
plt.tight_layout()
plt.show()

# Define the range for p and q
p_range = range(0, 7)  # Limit p to 0-7
q_range = range(0, 7)  # Limit q to 0-7
d = 1  # Set differencing term to 1 based on stationarity test

# Initialize variables to store the best parameters and AIC value
best_aic = float('inf')
best_order = None

# Iterate over all combinations of p and q to find the best ARIMA model
for p in p_range:
    for q in q_range:
        try:
            model = ARIMA(train_data['TEMP_F'], order=(p, d, q), enforce_stationarity=False, enforce_invertibility=False)
            results = model.fit()
            if results.aic < best_aic:
                best_aic = results.aic
                best_order = (p, d, q)
        except Exception as e:
            print(f"ARIMA({p},{d},{q}) model fitting failed: {e}")
            continue  # Skip this combination if the model fails to fit

print(f'\nBest ARIMA order: {best_order} with AIC: {best_aic}')

# Fit the ARIMA model with the best order (on the training set)
best_model = ARIMA(train_data['TEMP_F'], order=best_order, enforce_stationarity=False, enforce_invertibility=False)
best_arima_result = best_model.fit()

# Display the model summary
print(best_arima_result.summary())

# Define the number of steps to forecast
forecast_steps = len(test_data)  # Forecast the next 24 time steps

# Generate forecast
best_forecast = best_arima_result.get_forecast(steps=forecast_steps)
best_mean_forecast = best_forecast.predicted_mean
best_confidence_intervals = best_forecast.conf_int()

# Get the dates and actual temperatures from the test set
test_dates = test_data['DATE']
actual_future_temps = test_data['TEMP_F'].values

# Plot the forecast results with confidence intervals and actual temperatures
plt.figure(figsize=(15, 6))
plt.plot(test_dates, best_mean_forecast, color='red', marker='o', linestyle='-', label='Best Forecast')
plt.fill_between(test_dates,
                 best_confidence_intervals.iloc[:, 0],
                 best_confidence_intervals.iloc[:, 1], color='pink', alpha=0.3, label='Confidence Interval')

# Plot the actual temperatures
plt.plot(test_dates, actual_future_temps, color='blue', marker='x', linestyle='--', label='Actual Temperature')

plt.title('Best ARIMA Model Forecast vs Actual Temperatures')
plt.xlabel('Date')
plt.ylabel('Temperature (°F)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()





test_stationarity(train_data['TEMP_F'])

