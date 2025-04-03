import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from statsmodels.tsa.holtwinters import ExponentialSmoothing

dataset = pd.read_csv('Data/combined_data_2000-08-01_to_2025-01-01.csv', usecols=['Date', 'Close'], index_col='Date', parse_dates=True)

print(dataset.info())

dataset.index.freq = 'D'

print(dataset.info())

train_size = int(0.8*dataset.size)

train = dataset.iloc[:train_size]
test = dataset.iloc[train_size:]

model = ExponentialSmoothing(train['Close'], trend='mul', seasonal='mul', seasonal_periods=365).fit()

test[:30]['Close'].plot(label='Actual', figsize=(12, 9))
model.forecast(30).plot(label='Forecast')
plt.legend()
plt.title("30-Day Forecast vs Actual")
plt.ylabel("Closing Price")

os.makedirs("plots", exist_ok=True)

plt.savefig("plots/triple_exponential_smoothing.png")
plt.show()