import os
import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry

output_dir = "Data"
os.makedirs(output_dir, exist_ok=True)

def download_weather_data():
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
		"latitude": 52.52,
		"longitude": 13.41,
		"start_date": "2000-01-01",
		"end_date": "2025-03-24",
		"daily": ["weather_code", "shortwave_radiation_sum", "temperature_2m_mean", "temperature_2m_max", "temperature_2m_min", "sunshine_duration", "precipitation_sum", "precipitation_hours", "rain_sum", "daylight_duration", "snowfall_sum"],
		"hourly": "temperature_2m",
		"timezone": "America/New_York"
	}
    responses = openmeteo.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]
    print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
    print(f"Elevation {response.Elevation()} m asl")
    print(f"Timezone {response.Timezone()}{response.TimezoneAbbreviation()}")
    print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

                                # Process daily data. The order of variables needs to be the same as requested.
    daily = response.Daily()
    daily_weather_code = daily.Variables(0).ValuesAsNumpy()
    daily_shortwave_radiation_sum = daily.Variables(1).ValuesAsNumpy()
    daily_temperature_2m_mean = daily.Variables(2).ValuesAsNumpy()
    daily_temperature_2m_max = daily.Variables(3).ValuesAsNumpy()
    daily_temperature_2m_min = daily.Variables(4).ValuesAsNumpy()
    daily_sunshine_duration = daily.Variables(5).ValuesAsNumpy()
    daily_precipitation_sum = daily.Variables(6).ValuesAsNumpy()
    daily_precipitation_hours = daily.Variables(7).ValuesAsNumpy()
    daily_rain_sum = daily.Variables(8).ValuesAsNumpy()
    daily_daylight_duration = daily.Variables(9).ValuesAsNumpy()
    daily_snowfall_sum = daily.Variables(10).ValuesAsNumpy()

    daily_data = {"date": pd.date_range(
        start=pd.to_datetime(daily.Time(), unit="s", utc=True),
        end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True) - pd.Timedelta(seconds=daily.Interval()),
        freq=pd.Timedelta(seconds=daily.Interval())
    )}

    daily_data["weather_code"] = daily_weather_code
    daily_data["shortwave_radiation_sum"] = daily_shortwave_radiation_sum
    daily_data["temperature_2m_mean"] = daily_temperature_2m_mean
    daily_data["temperature_2m_max"] = daily_temperature_2m_max
    daily_data["temperature_2m_min"] = daily_temperature_2m_min
    daily_data["sunshine_duration"] = daily_sunshine_duration
    daily_data["precipitation_sum"] = daily_precipitation_sum
    daily_data["precipitation_hours"] = daily_precipitation_hours
    daily_data["rain_sum"] = daily_rain_sum
    daily_data["daylight_duration"] = daily_daylight_duration
    daily_data["snowfall_sum"] = daily_snowfall_sum

    daily_dataframe = pd.DataFrame(data = daily_data)
    print(daily_dataframe)
    
    output_path = os.path.join(output_dir, "weather_data_2000-01-01_to_2025-03-24.csv")
    daily_dataframe.to_csv(output_path, index=False)
    print(f"Weather data exported to {output_path}")

download_weather_data()