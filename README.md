# Weather Data Analysis

To begin, we explored various data sources including NOAA, Weather Underground, OpenWeatherMap, WeatherAPI, TimeAndDate, and Kaggle. We encountered two primary challenges in this phase:

1. **Data Source Accessibility:** Some of the chosen websites were not entirely free or offered limited data access without a subscription. This posed a constraint on data availability for the project.

2. **Data Structure Ambiguity:** Upon securing access to data through API calls, another hurdle emerged—the structure of the JSON files. Understanding the structure and meaning of certain columns was not straightforward, necessitating a substantial amount of time to decipher the data.

## Data Collection Approaches

To overcome these challenges, we used a combination of data collection methods:

1. **Direct Data Download:** In one instance, we directly downloaded a dataset from Kaggle. While this source provided a structured dataset, it still required preprocessing to meet project requirements.

2. **Web Scraping:** Additionally, we resorted to web scraping to extract data from NOAA. This required parsing and structuring the data for further analysis.

## Preliminary Visualizations of Data

Currently, we have trained an ARIMA model using temperature from October only, and the following are some initial visualizations of the data.
![Weather Data Visualization](./images/time_vs_temp.png)
Additionally, we have downloaded historical weather data.

## Detailed Description of Data Processing

The data processing was done to the historical data mainly and we are still considering if we are going to include it into our future model. The figure below shows that there is a time lag between the historical temperature and the scraped ground truth, and this data is fixed in data comparison between scraped data and historical data.

Also, there are some anomalies in the raw historical data. For example, a temperature of 99999. If the gap between two correct records is short, we simply use the average to fill the gap, but if the gap is large, we then choose to use autoregressive modeling to fill the blank.

## Data Modeling Methods Used So Far

We started with autoregressive modeling, specifically focusing on the AutoRegressive Integrated Moving Average (ARIMA) model. ARIMA is widely employed in short-term temperature prediction due to its ability to capture autocorrelation, trends, seasonality, and short-term fluctuations in time series data.

We began with stationarity assessment, and the autocorrelation and partial autocorrelation figures are shown below.

### Plots Analysis

The generated plots provided critical insights into the dataset's characteristics:

- **ACF Plot:** The autocorrelation function (ACF) plot exhibited a gradual tailing-off pattern, suggesting the potential benefit of incorporating a moving average (MA) term in the model.

- **PACF Plot:** Conversely, the partial autocorrelation function (PACF) plot displayed a sharp cut-off after approximately 2 lags. This indicated that an autoregressive (AR) term of around 2 might be suitable.

Based on this initial analysis, we selected the starting ARIMA model as ARIMA(2,0,1).

**Results:**

The results of this initial implementation demonstrated a close fit to the temperature trend over the forecasted period. However, a closer examination raised concerns. The predicted temperatures formed an almost straight-line pattern, raising doubts about the model's ability to hold up in other instances and scenarios.

### ARIMA Model Parameters: AR, I, MA

In the context of the ARIMA model, the selection of its parameters—AutoRegressive (AR), Integrated (I), and Moving Average (MA)—plays a pivotal role in the model's performance and predictive capabilities. We considered these parameters in-depth during our modeling process:

- **AR (AutoRegressive):** The AR parameter represents the relationship between an observation and a number of lagged observations, specifically, the previous time points. For this project, where we aimed to generate temperature predictions for 24 hours ahead, we anticipated a higher value for the AR parameter. A larger AR value would enable the model to incorporate more historical data into its predictions, capturing potential patterns and dependencies over a longer period.

- **I (Integrated):** The Integrated parameter addresses non-stationarity in the data. Given that the ADF test results confirmed the dataset's stationarity, we placed less emphasis on this parameter in our modeling approach. The stationary nature of the data reduced the need for differencing operations to achieve stationarity.

- **MA (Moving Average):** The MA parameter accounts for the relationship between an observation and a residual error from a moving average model applied to lagged observations. Our inclination was to increase the MA value in our ARIMA model. This decision was driven by the primary concern of potentially missing complex patterns in the data. By incorporating a higher MA value, the model could better capture and account for variations and fluctuations in the temperature data.

---

In our pursuit of optimizing the ARIMA model for temperature prediction, we decided to conduct a systematic exploration of various model orders. Specifically, we varied the parameters p (AR) and q (MA) within the range of 6 to assess their impact on the model's performance.

### Model Order Iteration

We initiated a programmatic iteration over a range of p and q values, ranging from 0 to 6. The aim was to identify the optimal combination of these parameters within an order range of (6, 0, 6).

## LSTM Model

As ARIMA is a relatively simple model and may fail to capture underlying trends influenced by factors beyond temperature, we plan to improve our predictions using a more comprehensive model like Long Short-Term Memory (LSTM) networks. LSTM is a type of recurrent neural network (RNN) that is particularly well-suited for capturing long-term dependencies and temporal relationships in time series data. This makes it ideal for our weather prediction task, where multiple features and their interactions over time must be considered.

We intend to leverage LSTM's ability to handle both short-term fluctuations and long-term trends to create a more robust model that can effectively account for various influencing factors. We will incorporate not only temperature but also additional weather-related variables such as humidity, wind speed, and atmospheric pressure to improve the predictive accuracy of the model.

Our next steps involve preparing the dataset for LSTM, which will include normalizing features, creating time windows for the input sequence, and splitting the data into training, validation, and testing sets. We will then build, train, and fine-tune the LSTM model to evaluate its performance in comparison to the ARIMA model.

## Preliminary Results

(TBD: Discuss initial model outputs, patterns found, and any insights gathered.)
