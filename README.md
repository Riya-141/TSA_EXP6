# Ex.No: 6               HOLT WINTERS METHOD
### Date: 21.11.2026
### AIM:
To implement the Holt-Winters Exponential Smoothing method on HDFC Bank stock price data to analyze monthly trends, evaluate model performance using RMSE, and predict future closing prices.
### DATASET:
HDFC bank stock price data.
### SOFTWARE REQUIRED/TOOL USED:
Google Colab
### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
```
```
# Load the dataset
data = pd.read_csv('/content/drive/MyDrive/Time Series/HDFCBANK.NS.csv')

data['Date'] = pd.to_datetime(data['Date'], format='mixed', dayfirst=True, errors='coerce')
data = data.dropna(subset=['Date'])
data.set_index('Date', inplace=True)

data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
data = data.dropna(subset=['Close'])
data = data.sort_index()
```
```
data.head()
```
```
# Resample data monthly
data_monthly = data['Close'].resample('MS').mean()
data_monthly = data_monthly.dropna()
data_monthly.head()
```
```
# Plot monthly data
data_monthly.plot()
plt.title('HDFC Bank Monthly Closing Price')
plt.xlabel('Month')
plt.ylabel('Closing Price')
plt.show()
```
```
# Scale the data
scaler = MinMaxScaler()

scaled_data = pd.Series(
    scaler.fit_transform(data_monthly.values.reshape(-1, 1)).flatten(),
    index=data_monthly.index
)

scaled_data.plot()
plt.title('Scaled Data Plot')
plt.xlabel('Month')
plt.ylabel('Scaled Closing Price')
plt.show()
```
```
# Seasonal decomposition
from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(data_monthly, model="additive", period=12)
decomposition.plot()
plt.show()
```
```
# Split test and train data
# Create a model using Holt-Winters method
# Train with train data and evaluate the model predictions against test data

# Scaled data plot
scaled_data = scaled_data + 1

train_data = scaled_data[:int(len(scaled_data) * 0.8)]
test_data = scaled_data[int(len(scaled_data) * 0.8):]

model_add = ExponentialSmoothing(
    train_data,
    trend='add',
    seasonal='mul',
    seasonal_periods=12
).fit()

test_predictions_add = model_add.forecast(steps=len(test_data))

ax = train_data.plot()
test_predictions_add.plot(ax=ax)
test_data.plot(ax=ax)

ax.legend(["train_data", "test_predictions_add", "test_data"])
ax.set_title('Visual Evaluation')
plt.show()
```
```
# Model performance metrics
print("Model performance metrics:")
print("RMSE:")
print(np.sqrt(mean_squared_error(test_data, test_predictions_add)))

print("Standard deviation and mean:")
print(np.sqrt(scaled_data.var()), scaled_data.mean())
```
```
# Create the final model and predict future data

final_model = ExponentialSmoothing(
    data_monthly,
    trend='add',
    seasonal='mul',
    seasonal_periods=12
).fit()

final_predictions = final_model.forecast(
    steps=int(len(data_monthly) / 4)
)
```
```
# Plot final prediction
ax = data_monthly.plot()
final_predictions.plot(ax=ax)

ax.legend(["data_monthly", "final_predictions"])
ax.set_xlabel('Number of monthly observations')
ax.set_ylabel('Closing Price')
ax.set_title('HDFC Bank Stock Price Prediction')
plt.show()

print("Final prediction:")
print(final_predictions)
```
## OUTPUT:

### Scaled_data plot:

<img width="747" height="515" alt="image" src="https://github.com/user-attachments/assets/ad69d00f-1191-43e0-b74c-ff6ee92b5adf" />

### Decomposed plot:

<img width="822" height="536" alt="image" src="https://github.com/user-attachments/assets/c3c5ed3f-16d8-4afa-97ea-475db00b1e17" />

### TEST_PREDICTION:

<img width="716" height="517" alt="image" src="https://github.com/user-attachments/assets/4a9e1ca1-5834-4c2d-ad98-404ca2ba335f" />

### Model performance metrics:

<img width="418" height="105" alt="image" src="https://github.com/user-attachments/assets/bc1125ec-38aa-43a4-b2ee-5a997a08796b" />

### FINAL_PREDICTION:

<img width="823" height="512" alt="image" src="https://github.com/user-attachments/assets/d4775ca1-c537-433f-bed9-2ed58d0e3939" />

### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
