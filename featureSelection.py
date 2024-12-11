import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import mutual_info_regression, RFE
from sklearn.ensemble import RandomForestRegressor
import warnings
def preprocess_data(input_file, sequence_length, scaler=None, fit_scaler=False, forecast_horizon=24):
    """
    Preprocess the data and create sequences and target variables.

    Parameters:
    - input_file: path to the input CSV file.
    - sequence_length: length of the input sequence.
    - scaler: fitted StandardScaler object for the test set.
    - fit_scaler: whether to fit the scaler on this data (training set only).
    - forecast_horizon: number of time steps to forecast (default 24 hours).

    Returns:
    - X, y: feature and target arrays (numpy).
    - scaler: fitted StandardScaler object (training set only).
    """
    data = pd.read_csv(input_file)
    # data=data[-365*4*24:]
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

    # Create 24-hour lag features for temperature
    for lag in range(1, 25):
        data[f'Temperature_lag_{lag}'] = data['Temperature'].shift(lag)

    # Drop rows with missing lag feature values
    data = data.dropna(subset=[f'Temperature_lag_{lag}' for lag in range(1, 25)])

    # Select final features
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

    # Create target variable: max temperature in the next 24 hours
    data['Max_Temp_24h'] = data['Temperature'].rolling(window=forecast_horizon).max().shift(-forecast_horizon + 1)

    # Drop rows where target cannot be computed
    data = data.dropna(subset=['Max_Temp_24h'])
    return data
# Compute correlation coefficients
warnings.filterwarnings('ignore')  # Turn off unnecessary warnings
data = preprocess_data('data/1.csv', 24, fit_scaler=True, forecast_horizon=24)
features = ['Temperature', 'Relative Humidity', 'Dwpt',
                      'Date_Sin', 'Date_Cos', 'Month_Sin', 'Month_Cos',
                      'Hour_Sin', 'Hour_Cos']
target = "Max_Temp_24h"
def correlation_analysis(data, features, target):
    correlation_matrix = data[features + [target]].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Feature Correlation Matrix')
    plt.show()

    target_correlation = correlation_matrix[target].drop(target).sort_values(ascending=False)
    print("Correlation with the target variable:")
    print(target_correlation)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=target_correlation.values, y=target_correlation.index, palette='viridis')
    plt.title('Correlation of Features with Target Variable')
    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Features')
    plt.show()

# 1.2 Mutual Information
def mutual_information_analysis(data, features, target):
    X = data[features]
    y = data[target]

    mi = mutual_info_regression(X, y, random_state=42)
    mi_series = pd.Series(mi, index=features).sort_values(ascending=False)

    print("Mutual information scores of features:")
    print(mi_series)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=mi_series.values, y=mi_series.index, palette='magma')
    plt.title('Mutual Information of Features')
    plt.xlabel('Mutual Information Score')
    plt.ylabel('Features')
    plt.show()

# 1.3 Tree-based feature importance
def tree_based_feature_importance(data, features, target):
    X = data[features]
    y = data[target]

    X_train_rf, X_val_rf, y_train_rf, y_val_rf = train_test_split(
        X, y, test_size=0.2, shuffle=False, random_state=42
    )

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train_rf, y_train_rf)

    importances = rf.feature_importances_
    feature_importance = pd.Series(importances, index=features).sort_values(ascending=False)

    print("Feature importance based on Random Forest:")
    print(feature_importance)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importance.values, y=feature_importance.index, palette='Spectral')
    plt.title('Feature Importances from Random Forest')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.show()

# 1.4 Recursive Feature Elimination (RFE)
def rfe_feature_selection(data, features, target, n_features_to_select=5):
    X = data[features]
    y = data[target]

    X_train_rfe, X_val_rfe, y_train_rfe, y_val_rfe = train_test_split(
        X, y, test_size=0.2, shuffle=False, random_state=42
    )

    model = LinearRegression()
    rfe = RFE(estimator=model, n_features_to_select=n_features_to_select)
    rfe.fit(X_train_rfe, y_train_rfe)

    ranking = pd.Series(rfe.ranking_, index=features).sort_values()

    print("Feature rankings from RFE:")
    print(ranking)

    selected_features = ranking[ranking == 1].index.tolist()
    print(f"Selected features: {selected_features}")

    plt.figure(figsize=(10, 6))
    sns.barplot(x=ranking.values, y=ranking.index, palette='Blues_d')
    plt.title('Feature Rankings from RFE')
    plt.xlabel('Ranking')
    plt.ylabel('Features')
    plt.show()

    return selected_features

# Execute feature importance evaluation
correlation_analysis(data, features, target)
mutual_information_analysis(data, features, target)
tree_based_feature_importance(data, features, target)
selected_features_rfe = rfe_feature_selection(data, features, target, n_features_to_select=5)

# Choose the features selected by RFE for further analysis
print(f"Features selected by RFE: {selected_features_rfe}")

# 2. Use the selected features for model training and evaluation
# Update feature list
selected_features = selected_features_rfe  # Or select as needed
