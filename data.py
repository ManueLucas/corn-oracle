import yfinance as yf
import pandas as pd
import numpy as np
import os
#from datetime import datetime

# Define the ticker symbol for corn futures (ZC=F for Corn Futures)
ticker = "ZC=F"

# Define the start and end dates for the data
data_start = "2000-08-01"
data_end = "2025-01-01"
#data_end = datetime.today().strftime('%Y-%m-%d')

# Download historical OHLC data using yfinance
def download_corn_futures_data(ticker, start_date, end_date):
    try:
        # Fetch the data
        data = yf.download(ticker, start=start_date, end=end_date, progress=True)
                                    
        # Check if data is returned
        if data.empty:
            print(f"No data found for {ticker}.")
            return None
        
        # Flatten multi-level column index if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        # Make the index the date
        data.index = pd.to_datetime(data.index)
        full_index = pd.date_range(start=data.index.min(), end=data.index.max(), freq="D")
        # data = data.reindex(full_index)
        
        # Use rolling Statistics imputation to fill out unrecorded data
        data["Open"] = data["Open"].fillna(method="ffill").fillna(method="bfill").fillna(data["Open"].rolling(5, min_periods=1).mean())
        data["High"] = data["High"].fillna(method="ffill").fillna(method="bfill").fillna(data["High"].rolling(5, min_periods=1).mean())
        data["Low"] = data["Low"].fillna(method="ffill").fillna(method="bfill").fillna(data["Low"].rolling(5, min_periods=1).mean())
        data["Close"] = data["Close"].fillna(method="ffill").fillna(method="bfill").fillna(data["Close"].rolling(5, min_periods=1).mean())
        data["Volume"] = data["Volume"].fillna(0)
        
        data["MA_7"] = data["Close"].rolling(window=7, min_periods=1).mean()   # 7-day rolling moving average
        data["MA_30"] = data["Close"].rolling(window=30, min_periods=1).mean() # 30-day rolling moving average

        os.makedirs("Data", exist_ok=True)
        
        # Save to a CSV file
        output_file = os.path.join("Data", f"corn_futures_{start_date}_to_{end_date}.csv")
        data.to_csv(output_file, index_label='Date')
        print(f"Data downloaded and saved to {output_file}")

        return data
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Run the function
data = download_corn_futures_data(ticker, data_start, data_end)

# If needed, display the first few rows
if data is not None:
    print(data.head())
