import yfinance as yf
import pandas as pd
import numpy as np
import os
from weather_data import download_weather_data
from sklearn.model_selection import train_test_split
#from datetime import datetime

# Define the ticker symbol for corn futures (ZC=F for Corn Futures)
ticker = "ZC=F"

# Define the start and end dates for the data
start_date = "2000-08-01"
corn_end_date = "2025-01-01"
weather_end_date = "2024-12-31"
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
        data = data.reindex(full_index)
        
        # Use rolling Statistics imputation to fill out unrecorded data
        data["Open"] = data["Open"].fillna(method="ffill").fillna(method="bfill").fillna(data["Open"].rolling(5, min_periods=1).mean())
        data["High"] = data["High"].fillna(method="ffill").fillna(method="bfill").fillna(data["High"].rolling(5, min_periods=1).mean())
        data["Low"] = data["Low"].fillna(method="ffill").fillna(method="bfill").fillna(data["Low"].rolling(5, min_periods=1).mean())
        data["Close"] = data["Close"].fillna(method="ffill").fillna(method="bfill").fillna(data["Close"].rolling(5, min_periods=1).mean())
        data["Volume"] = data["Volume"].fillna(0)
        
        data["MA_7"] = data["Close"].rolling(window=7, min_periods=1).mean()   # 7-day rolling moving average
        data["MA_30"] = data["Close"].rolling(window=30, min_periods=1).mean() # 30-day rolling moving average

        os.makedirs("Data", exist_ok=True)
        
        
        print(f"len(data): {len(data)}")
        
        # Save to a CSV file
        output_file = os.path.join("Data", f"corn_futures_{start_date}_to_{end_date}.csv")
        
        data.to_csv(output_file, index_label='Date')

        print(f"Data downloaded and saved to {output_file}")

        return data
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
def combine_dataframes(ticker, start_date, corn_end_date, weather_end_date):
    corn_data = download_corn_futures_data(ticker, start_date, corn_end_date)
    weather_data = download_weather_data(start_date, weather_end_date)

    # Normalize both indexes to be timezone-naive
    corn_data.index = pd.to_datetime(corn_data.index).tz_localize(None)
    weather_data['date'] = pd.to_datetime(weather_data['date']).dt.tz_localize(None)
    
    weather_data.rename(columns={'date': 'Date'}, inplace=True)
    weather_data.set_index('Date', inplace=True)

    combined_data = pd.concat([corn_data, weather_data], axis=1)

    output_file = os.path.join("Data", f"combined_data_{start_date}_to_{corn_end_date}.csv")
    combined_data.to_csv(output_file, index_label='Date')

    print(f"Data downloaded and saved to {output_file}")

    return combined_data
    
    
def download_corn_futures_full_data(start_date = '2024-08-01', end_date = '2025-01-01'):
    try:
        # Fetch the data
        data = yf.download('ZC=F', start=start_date, end=end_date, progress=True)
                                    
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
        data = data.reindex(full_index)
        
        # Use rolling Statistics imputation to fill out unrecorded data
        data["Open"] = data["Open"].fillna(method="ffill").fillna(method="bfill").fillna(data["Open"].rolling(5, min_periods=1).mean())
        data["High"] = data["High"].fillna(method="ffill").fillna(method="bfill").fillna(data["High"].rolling(5, min_periods=1).mean())
        data["Low"] = data["Low"].fillna(method="ffill").fillna(method="bfill").fillna(data["Low"].rolling(5, min_periods=1).mean())
        data["Close"] = data["Close"].fillna(method="ffill").fillna(method="bfill").fillna(data["Close"].rolling(5, min_periods=1).mean())
        data["Volume"] = data["Volume"].fillna(0)
        
        data["MA_7"] = data["Close"].rolling(window=7, min_periods=1).mean()   # 7-day rolling moving average
        data["MA_30"] = data["Close"].rolling(window=30, min_periods=1).mean() # 30-day rolling moving average
        os.makedirs("Data", exist_ok=True)
        
        output_file = os.path.join("Data", f"corn_futures_{start_date}_to_{end_date}.csv")
        
        data.to_csv(output_file, index_label='Date')

        print(f"Data downloaded and saved to {output_file}")
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
    return data

def download_corn_futures_eval_data():
    start_date = '2025-01-01'
    end_date = '2025-03-21'
    try:
        # Fetch the data
        data = yf.download('ZC=F', start=start_date, end=end_date, progress=True)
                                    
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
        data = data.reindex(full_index)
        
        # Use rolling Statistics imputation to fill out unrecorded data
        data["Open"] = data["Open"].fillna(method="ffill").fillna(method="bfill").fillna(data["Open"].rolling(5, min_periods=1).mean())
        data["High"] = data["High"].fillna(method="ffill").fillna(method="bfill").fillna(data["High"].rolling(5, min_periods=1).mean())
        data["Low"] = data["Low"].fillna(method="ffill").fillna(method="bfill").fillna(data["Low"].rolling(5, min_periods=1).mean())
        data["Close"] = data["Close"].fillna(method="ffill").fillna(method="bfill").fillna(data["Close"].rolling(5, min_periods=1).mean())
        data["Volume"] = data["Volume"].fillna(0)
        
        data["MA_7"] = data["Close"].rolling(window=7, min_periods=1).mean()   # 7-day rolling moving average
        data["MA_30"] = data["Close"].rolling(window=30, min_periods=1).mean() # 30-day rolling moving average
        os.makedirs("Data", exist_ok=True)
        
        eval_output_file = os.path.join("Data", f"eval_corn_futures_{start_date}_to_{end_date}.csv")
        
        data.to_csv(eval_output_file, index_label='Date')

        print(f"Data downloaded and saved to {eval_output_file}")
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
    return data

def prepare_sequences_targets(data, sequence_length):
            """
            Prepares sequences (fixed window of historical data) and corresponding targets for training and testing.

            Args:
                data: numpy array of shape (num_samples, num_features).
                sequence_length: length of input sequences.

            Returns:
                sequences: numpy array of shape (num_samples, sequence_length, input_size).
                targets: numpy array of shape (num_samples, output_size).
            """
            print(data.shape)
            num_samples = data.shape[0]
            sequences = []
            targets = []

            for i in range(num_samples - sequence_length): #
                seq = data[i:i + sequence_length]
                target = data[i + sequence_length]  # the next time step
                sequences.append(seq)
                targets.append(target)

            sequences = np.array(sequences)  # shape: (num_samples, sequence_length, input_size)
            targets = np.array(targets)       # shape: (num_samples, output_size)

            return sequences, targets


combine_dataframes(ticker, start_date, corn_end_date, weather_end_date)
# # Run the function
# data = download_corn_futures_data(ticker, data_start, data_end)

# # If needed, display the first few rows
# if data is not None:
#     print(data.head())

#data eval test
# data = download_corn_futures_eval_data()

# # If needed, display the first few rows
# if data is not None:
#     print(data.head())



