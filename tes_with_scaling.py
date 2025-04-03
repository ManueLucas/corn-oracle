import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from statsmodels.tsa.holtwinters import ExponentialSmoothing

dataset = pd.read_csv('Data/combined_data_2000-08-01_to_2025-01-01.csv', usecols=['Date', 'Close'], index_col='Date', parse_dates=True)

print(dataset.info())

dataset.index.freq = 'D'

print(dataset.info())

# to prevent overflow
scale_factor = 1000.0


train_size = int(0.8 * len(dataset)) 

train = dataset.iloc[:train_size]
test = dataset.iloc[train_size:]

# Scale the values
train_scaled = train['Close'] / scale_factor

model = ExponentialSmoothing(train_scaled, trend='mul', seasonal='mul', seasonal_periods=365).fit()

# Forecast and scale back up
forecast_scaled = model.forecast(30)
forecast = forecast_scaled * scale_factor

# Plots n dat
test[:30]['Close'].plot(label='Actual', figsize=(12, 9))
forecast.plot(label='Forecast')
plt.legend()
plt.title("30-Day Forecast vs Actual")
plt.ylabel("Closing Price")

os.makedirs("plots", exist_ok=True)

plt.savefig("plots/triple_exponential_smoothing_with_scaling.png")
plt.show()