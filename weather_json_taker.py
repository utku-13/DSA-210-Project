import requests
import json
# Define API parameters
api_key = "6d06df436a30403eb6a131519252103"  # Replace with your actual API key
location = "Istanbul,Turkey"  # Kadıköy (Asian side of Istanbul)
start_date = "2025-01-01"
end_date = "2025-03-15"
format_type = "json"  # Response format

# API Endpoint
url = f"https://api.worldweatheronline.com/premium/v1/past-weather.ashx"

# Request parameters
params = {
    "key": api_key,
    "q": location,  # Location (Kadıköy, Istanbul)
    "date": start_date,
    "enddate": end_date,
    "tp": "24",  # Time interval (daily data)
    "format": format_type
}

# Make API request
response = requests.get(url, params=params)

# Check response
if response.status_code == 200:
    weather_data = response.json()
    print(weather_data)
else:
    print(f"Error: {response.status_code}, {response.text}")

with open("weather_data1.json", "w") as file:
    json.dump(weather_data, file, indent=4)

print("Weather data has been written to 'weather_data.json'.")