## Project Overview

### Goal
Our primary goal is to predict **tomorrow's highest temperature** based on **today's weather conditions**, such as temperature, humidity, and wind speed. By collecting historical weather data and training regression models, we aim to generalize well to new data for accurate forecasting.

### Data Collection
We plan to gather historical weather data, including **daily high temperatures** and other relevant conditions, from reputable sources like the **National Oceanic and Atmospheric Administration (NOAA)** and **Weather Underground**. Data will be obtained via **APIs or web scraping**, and we will ensure consistency and accuracy during preprocessing.

### Modeling Approach
We will begin with **autoregression models**, which leverage past data to forecast future values. Depending on time and progress, we will also explore more complex models like **Long Short-Term Memory (LSTM)** networks. We may use an **ensemble approach** to combine multiple models for better performance.

### Data Visualization
Data visualization is critical for understanding trends. We will use:

- **Time Series Plots** to highlight temperature trends and seasonality.
- **Scatter Plots & Correlation Matrices** to assess relationships between weather variables.
- **Interactive Visualizations** (e.g., Plotly) for deep analysis of specific time ranges and conditions.
- **Error Plots** comparing predicted vs. actual temperatures for model performance evaluation.
- **Moving Average Plots** to visualize model alignment with real trends.

### Test Plan
The model will undergo three phases of testing:

1. **Historical Data Testing**  
   A **20% hold-out test set** will be reserved during the initial development phase to ensure the model can generalize to unseen data.

2. **Real-time Testing**  
   The model will be used to **predict tomorrow's temperature**, which will be compared with actual values once available. This iterative process will help refine the model over time.

3. **Cross-Location Testing**  
   To evaluate generalizability, we will apply the model to temperature data from **different cities** not included in the training dataset. This approach will test the model's robustness across various geographical regions and climates.
