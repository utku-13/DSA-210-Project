# Working Hour Analysis

---

**Motivation and Project Idea**

---


* I have observed that my coffee consumption might be influenced by different external factors, such as weather conditions and academic events (like exam weeks or project deadlines). I suspect that on cold or rainy days, I may consume more coffee, or that stressful periods such as exams could lead to increased coffee intake. Through this project, I plan to investigate these correlations and try to understand that my coffee consumption is highly dependent on my daily life.

---

**Dataset Information**

---

* For Coffee Consumption:
I will track my daily coffee/tea consumption manually in a spreadsheet, noting the total number of cups each day.

* For Weather (Temperature, Precipitation, Cloud Cover):
I will fetch weather data using a free API. This data will include daily temperature readings, precipitation levels, and cloud cover percentages. The API response will be processed and stored in CSV format.

* For Academic Events (Exam/Submission Weeks):
I will fetch events from the calendar on my iPhone via an API. Specifically, I will mark exam weeks, assignment deadlines, or any other significant event periods. This information will be stored in CSV format, indicating the dates and nature of events.
---

**Data Process**

---

1. Raw Data Collection:
Coffee Consumption: I will maintain a simple daily log (manually) in a spreadsheet, later exporting it to a CSV file.
Weather Data: Retrieved via an online API in JSON format. I will parse out relevant fields (temperature, precipitation, cloud cover) and store them in a CSV file.
Event Data (Exam, Submissions, etc.): Fetched from my iPhone calendar API. I will parse the event dates and types, then export the information to CSV.

2. Conversion and Cleaning:
The manually recorded spreadsheet for coffee consumption will be converted to CSV.
Weather JSON data will be parsed using Python to extract temperature, precipitation, and cloud cover, then saved to a CSV file.
Calendar events will be cleaned to ensure only relevant events (e.g., exams, submissions) are kept. All data sets will use consistent date formats for easier merging.

3. Data Integration:x
The processed CSV files will be merged based on the date fields (i.e., each day’s coffee consumption, weather data, and any events). This will result in a comprehensive daily dataset linking coffee intake with weather and event information.

---

**Hypothesis**

---

* Null Hypothesis:
There is no significant relationship between coffee consumption, weather conditions (temperature, precipitation, cloud cover), and academic events (exams, submission deadlines). Variations in daily coffee intake are purely random and not influenced by external factors.

* Alternative Hypothesis:
There is a significant relationship between coffee consumption, weather conditions, and events. Specifically, lower temperatures, rainy or cloudy weather, and stressful event periods (e.g., exam weeks) correspond to higher daily coffee intake.





