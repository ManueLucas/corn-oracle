import yfinance as yf
import pandas as pd

# Define the ticker symbol for corn futures (ZC=F for Corn Futures)
ticker = "ZC=F"

# Define the start and end dates for the data
data_start = "2000-07-17"
data_end = "2025-01-20"  # Update to the current date dynamically if needed

# Download historical OHLC data using yfinance
def download_corn_futures_data(ticker, start_date, end_date):
    try:
        # Fetch the data
        data = yf.download(ticker, start=start_date, end=end_date, progress=True)
        
        # Check if data is returned
        if data.empty:
            print(f"No data found for {ticker}.")
            return None

        # Save to a CSV file
        output_file = f"corn_futures_{start_date}_to_{end_date}.csv"
        data.to_csv(output_file)
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
