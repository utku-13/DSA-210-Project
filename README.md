# Coffee Consumption & External Factors

---

**Motivation and Project Idea**

---


* I have observed that my coffee consumption might be influenced by different external factors, such as weather conditions, academic events (like exam weeks or project deadlines), and even how much I sleep. I suspect that on cold or rainy days, I may consume more coffee, or that stressful periods such as exams could lead to increased coffee intake. Additionally, I want to examine whether fewer hours of sleep might also correlate with higher coffee intake the next day. Through this project, I plan to investigate these correlations and try to understand how my coffee consumption is tied to my daily life.


---

**Dataset Information**

---
* For Coffee Consumption:
I will track my daily coffee/tea consumption manually in a spreadsheet, noting the total number of cups each day.

* For Weather (Temperature, Precipitation, Cloud Cover):
I will fetch weather data using a free API. This data will include daily temperature readings, precipitation levels, and cloud cover percentages. The API response will be processed and stored in CSV format.

* For Academic Events (Exam/Submission Weeks):
I will fetch events from the calendar on my iPhone via an API. Specifically, I will mark exam weeks, assignment deadlines, or any other significant event periods. This information will be stored in CSV format, indicating the dates and nature of events.

* For Sleep (Daily Sleep Hours):
I will manually record my daily sleep hours in a spreadsheet. This data will then be converted to CSV format.
---

**Data Process**

---

1. Raw Data Collection:
Coffee Consumption: I will maintain a simple daily log (manually) in a spreadsheet, later exporting it to a CSV file.
Weather Data: Retrieved via an online API in JSON format. I will parse out relevant fields (temperature, precipitation, cloud cover) and store them in a CSV file.
Event Data (Exam, Submissions, etc.): Fetched from my iPhone calendar API. I will parse the event dates and types, then export the information to CSV.
Sleep Data: Sleep Data: Recorded manually in a spreadsheet, indicating total hours of sleep each night, and later converted to CSV.

2. Conversion and Cleaning:
The manually recorded spreadsheet for coffee consumption will be converted to CSV.
Weather JSON data will be parsed using Python to extract temperature, precipitation, and cloud cover, then saved to a CSV file.
Calendar events will be cleaned to ensure only relevant events (e.g., exams, submission deadlines) are kept.
The sleep spreadsheet will be reviewed for consistency (e.g., no overlapping days or missing entries) before final conversion to CSV.
All datasets will use consistent date formats for easier merging.

3. Data Integration:
The processed CSV files will be merged based on the date fields. This will result in a comprehensive daily dataset linking coffee intake with weather conditions, academic events, and sleep duration.

---

**Hypothesis**

---

* Null Hypothesis:
There is no significant relationship between coffee consumption, weather conditions (temperature, precipitation, cloud cover), academic events (exams, submission deadlines), or sleep duration. Variations in daily coffee intake are purely random and not influenced by these external factors.

* Alternative Hypothesis:
There is a significant relationship between coffee consumption, weather conditions, academic events, and sleep duration. Specifically, lower temperatures, rainy or cloudy weather, stressful event periods (e.g., exam weeks), and shorter sleep correlate with higher daily coffee intake. 

---

# Results and Data Analysis

## Findings

Various analyses were conducted to test the project's main hypotheses. This section presents the findings and results of these hypotheses.

### Correlation Between Variables

![Correlation Matrix](final_analysis_output/korelasyon_matrisi.png)

The correlation matrix shows the strength of relationships between coffee consumption and other variables. The strongest relationship in the matrix is a negative correlation between coffee consumption and sleep duration (-0.73). This result strongly supports our hypothesis of "less sleep, more coffee consumption." Weaker relationships were observed between weather variables and coffee consumption.

### Weather and Coffee Consumption

#### Temperature and Coffee Consumption

![Temperature and Coffee Relationship](final_analysis_output/sicaklik_kahve_iliskisi.png)

This graph showing the relationship between temperature and coffee consumption partially supports our hypothesis of "lower temperature, higher coffee consumption." The correlation value was calculated as -0.21, indicating that as temperature decreases, coffee consumption slightly increases, although this relationship is not very strong.

#### Precipitation and Coffee Consumption

![Precipitation and Coffee Relationship](final_analysis_output/yagis_kahve_iliskisi.png)

This box plot showing coffee consumption by precipitation categories does not clearly support our hypothesis of "higher coffee consumption on rainy days." The graph shows no significant difference in coffee consumption across precipitation categories.

#### Cloud Cover and Coffee Consumption

![Cloud Cover and Coffee Relationship](final_analysis_output/bulut_kahve_iliskisi.png)

This box plot showing coffee consumption by cloud cover categories does not support our hypothesis of "higher coffee consumption on cloudy/overcast days." No significant difference in coffee consumption is observed across different cloud cover categories.

### Academic Events and Coffee Consumption

![Academic Events and Coffee Relationship](final_analysis_output/akademik_kahve_iliskisi.png)

This graph comparing coffee consumption on exam/assignment submission days versus normal days contradicts our hypothesis of "higher coffee consumption during stressful academic periods." According to the graph, the average coffee consumption on exam/assignment days (1.40 cups) is lower than on normal days (1.61 cups). However, according to statistical analysis results (p-value: 0.6021), this difference is not statistically significant.

### Sleep and Coffee Consumption

#### Sleep Duration and Coffee Consumption

![Sleep Duration and Coffee Relationship](final_analysis_output/uyku_kahve_iliskisi.png)

This graph showing the relationship between sleep duration and coffee consumption strongly supports our hypothesis of "less sleep, more coffee consumption." The correlation value of -0.73 indicates that as sleep duration decreases, coffee consumption significantly increases. This stands out as the most influential factor affecting coffee consumption among all variables.

#### Sleep Categories and Coffee Consumption

![Sleep Categories and Coffee Relationship](final_analysis_output/uyku_kategorisi_kahve_iliskisi.png)

This box plot showing coffee consumption by sleep duration categories supports our hypothesis. Coffee consumption is notably higher in the insufficient sleep (<6 hours) category, and decreases as sleep duration increases.

#### Previous Day's Sleep Duration and Today's Coffee Consumption

![Previous Day's Sleep and Coffee Relationship](final_analysis_output/onceki_gun_uyku_kahve_iliskisi.png)

This graph showing the relationship between the previous day's sleep duration and today's coffee consumption supports another aspect of our hypothesis. Although the correlation value is not high (-0.15), the general trend indicates that a person who slept less the previous night tends to consume more coffee the next day.

### Combined Analysis of All Factors

#### Time Series Analysis

![Time Series of All Variables](final_analysis_output/tum_degiskenler_zaman_serisi.png)

This time series graph shows how all variables change over time. The exam/assignment days marked with red dashed lines in the graph represent the academic stress periods mentioned in our hypothesis. When examining the graph, the inverse relationship between sleep duration and coffee consumption is clearly visible.

#### Most Influential Factors

![Two Most Influential Factors](final_analysis_output/en_etkili_faktorler_3d.png)

This three-dimensional graph showing the effect of the two most influential factors (sleep duration and weekday/weekend) on coffee consumption based on multiple regression analysis supports some of our hypotheses. The dominant effect of sleep duration on coffee consumption (-1.0707) is clearly visible.

## Hypothesis Test Results

### Null Hypothesis

There is no significant relationship between coffee consumption and weather conditions, academic events, and sleep duration. Changes in daily coffee consumption are completely random and not affected by these external factors.

### Alternative Hypothesis

There is a significant relationship between coffee consumption and weather conditions, academic events, and sleep duration. Specifically, there is a correlation between lower temperatures, rainy/cloudy weather, stressful periods (exam weeks), and less sleep with higher coffee consumption.

### Conclusion

As a result of the analyses, our alternative hypothesis has been partially confirmed:

1. **Sleep and Coffee Consumption**: 
   - Hypothesis was **strongly confirmed**. 
   - A strong negative correlation (-0.73) was found between sleep duration and coffee consumption.
   - It emerged as the most influential factor in regression analysis (-1.0707 coefficient).

2. **Weather Conditions and Coffee Consumption**:
   - Hypothesis was **partially confirmed**.
   - A weak negative correlation (-0.21) was found between temperature and coffee consumption.
   - The hypothesis was not confirmed for precipitation and cloud cover variables.

3. **Academic Events and Coffee Consumption**:
   - Hypothesis was **not confirmed**.
   - Contrary to expectations, lower coffee consumption was observed on exam/assignment days.
   - This difference was not statistically significant (p=0.6021).

In conclusion, it was determined that sleep duration is the most decisive factor affecting coffee consumption, weather conditions have a limited effect, and academic stress factors did not show the expected effect.

## Future Work

In future studies, it could be beneficial to expand the analyses with more detailed measurements of academic stress factors (such as stress level surveys), longer data collection periods, and a larger participant pool.