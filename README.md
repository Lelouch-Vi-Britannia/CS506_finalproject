goal(s)
Our primary goal is to successfully predict tomorrow's highest temperature based on today's weather conditions (such as temperature, humidity, wind speed, etc.). This will be accomplished by collecting historical weather data and training models that can generalize well to new data.

Data Collection
We will collect historical weather data, including daily high temperatures and other relevant weather conditions, from reputable sources such as the National Oceanic and Atmospheric Administration (NOAA) and Weather Underground. The data will be scraped or accessed via available APIs, and we will process it to ensure consistency and accuracy.

Modeling Approach
We plan to begin with regression models, specifically autoregression, as it directly utilizes past data to make future predictions. Depending on the progress and time availability, we will also explore more complex models such as Long Short-Term Memory (LSTM) networks and other deep learning models. An ensemble approach may be used to combine different models and optimize the overall performance.

Data Visualization
We will visualize the collected and processed data using time series plots to highlight trends and seasonality in temperature. Scatter plots and correlation matrices will also be used to analyze relationships between different weather features (e.g., humidity, wind speed) and the target variable (temperature). Interactive visualizations, such as plotly charts, can be used for deeper analysis, allowing us to explore specific time ranges and conditions. Additionally, error plots comparing the predicted and actual temperatures will help assess the model performance.

Test Plan
During the development phase, the model will be tested in two ways:
Real-time Testing: As the model is developed, we will use it to predict tomorrow’s highest temperature and compare it with the actual temperature once it is available. This will allow us to iteratively refine the model based on its real-world performance.

Cross-Location Testing: To further evaluate the generalizability of the model, we will use it to predict historical temperature data from different cities that were not included in the training data. This will help assess how well the model adapts to different geographical regions and climates, ensuring it’s not overfitted to a specific location's weather patterns.
