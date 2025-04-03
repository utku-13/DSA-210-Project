import json
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates

weather_json_path = "/Users/utkuozer/Desktop/DSA Proje/DSA-210-Project/weather_data.json"

with open(weather_json_path, "r") as file:
    weather_data = json.load(file)  # Already a dictionary

weather_list = weather_data['data']['weather']

dates = []
avg_temps = []
cloud_covers = []
precipitations = []  # Using 'precipMM' as the rainfall indicator

for day in weather_list:
    dates.append(day['date'])
    avg_temps.append(float(day['avgtempC']))
    # Assume we use the first hourly entry for daily values.
    cloud_covers.append(float(day['hourly'][0]['cloudcover']))
    precipitations.append(float(day['hourly'][0]['precipMM']))

# Convert dates to pandas datetime objects for proper plotting
dates = pd.to_datetime(dates)

# Create subplots for each weather element
fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

# Plot average temperature
axes[0].plot(dates, avg_temps, marker='o', linestyle='-', label="Avg Temp (°C)")
axes[0].set_ylabel("Avg Temp (°C)")
axes[0].set_title("Weather Data Over 2 Months")
axes[0].legend()
axes[0].grid(True)

# Plot cloud cover
axes[1].plot(dates, cloud_covers, marker='s', linestyle='-', color='gray', label="Cloud Cover (%)")
axes[1].set_ylabel("Cloud Cover (%)")
axes[1].legend()
axes[1].grid(True)

# Plot precipitation (rain)
axes[2].plot(dates, precipitations, marker='^', linestyle='-', color='blue', label="Precipitation (mm)")
axes[2].set_ylabel("Precipitation (mm)")
axes[2].set_xlabel("Date")
axes[2].legend()
axes[2].grid(True)

# Set x-axis tick locator to display every day
for ax in axes:
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()