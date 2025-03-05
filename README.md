# Working Hour Analysis

---

**Motivation and Project Idea**
I have noticed that my daily schedule varies with both my sleep patterns and the weather. In particular, my working hours—i.e., the time I actively spend on tasks each day—seem to shift based on how well I slept the night before or the prevailing weather conditions (such as temperature, cloudiness, and precipitation). Through this project, I plan to investigate these factors together to see if my productivity can be optimized or if certain environmental or personal conditions might hinder my ability to work effectively.

---

**Dataset Information**

---

* For My Working Hours:
I will manually track my working hours in an Excel file that I maintain myself. This file will later be converted into a CSV format for analysis.

* For My Sleeping Hours:
I will attempt to track my sleeping hours using the Apple Health application's sleep data. If the fetched data is inaccurate or contains errors, I will switch to manually recording my sleep hours in an Excel file and convert the data to CSV.

* For Weather (Temperature, Precipitation, Cloud Cover):
I will fetch weather data from an online API. This data will include daily temperature readings, precipitation levels, and cloud cover percentages. The API response will be processed and stored in CSV format.

---

**Data Process**

---

1. Raw Data Collection:

Working Hours: Manually entered into Excel.
Sleeping Hours: Fetched from Apple Health in XML format (and manually recorded if needed).
Weather Data: Retrieved via an online API in JSON format.

2. Conversion and Cleaning:

Excel files will be exported as CSVs.
XML and JSON files will be parsed using Python to extract relevant fields, which will then be saved as CSV files.
Data cleaning will involve standardizing date formats, handling missing values, and converting categorical data into appropriate formats.

3. Data Integration:

Processed CSV files will be merged based on matching date fields to form a comprehensive dataset that includes working hours, sleeping hours, and corresponding weather data.

4. Storage and Version Control:

The final processed data files (CSV) will be stored in a designated data/ directory.

---

**Hypothesis**

---

* Null Hypothesis:
There is no significant relationship between sleeping hours, weather conditions, and working hours. In other words, variations in sleep duration and changes in weather (temperature, precipitation, and cloud cover) do not significantly affect the amount of time I spend working each day.

* Alternative Hypothesis:
There is a significant relationship between sleeping hours, weather conditions, and working hours. Specifically, differences in sleep duration and adverse weather conditions (such as low temperatures, heavy precipitation, or increased cloud cover) are associated with variations in my daily working hours.





