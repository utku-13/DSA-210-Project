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

Various statistical analyses were conducted to test the project's main hypotheses. This section presents the results of statistical tests and findings for each hypothesis.

### Correlation Between Variables

![Correlation Matrix](final_analysis_output/korelasyon_matrisi.png)

The correlation matrix shows the strength of relationships between coffee consumption and other variables. Statistically significant correlations are as follows:

- Coffee consumption and sleep duration: r = -0.73, p < 0.001 (highly statistically significant)
- Coffee consumption and temperature: r = -0.21, p = 0.043 (statistically significant)
- Coffee consumption and previous day's sleep duration: r = -0.15, p = 0.152 (not statistically significant)

According to these results, the null hypothesis is rejected for our "less sleep, more coffee consumption" hypothesis. Similarly, the null hypothesis is rejected for our "lower temperature, higher coffee consumption" hypothesis, although this relationship is weaker.

### Weather and Coffee Consumption

#### Temperature and Coffee Consumption

![Temperature and Coffee Relationship](final_analysis_output/sicaklik_kahve_iliskisi.png)

Pearson correlation test was applied to test the relationship between temperature and coffee consumption:
- r = -0.21, p = 0.043
- 95% confidence interval: [-0.39, -0.01]

These results show that coffee consumption slightly increases as temperature decreases. Since p < 0.05, the null hypothesis is rejected, and there is a statistically significant relationship between temperature and coffee consumption.

#### Precipitation and Coffee Consumption

![Precipitation and Coffee Relationship](final_analysis_output/yagis_kahve_iliskisi.png)

ANOVA test was applied to test the difference in coffee consumption across precipitation categories:
- F-value = 0.842, p = 0.476
- Effect size (eta-squared) = 0.031

Since p > 0.05, the null hypothesis cannot be rejected. There is no statistically significant difference in coffee consumption across precipitation categories.

#### Cloud Cover and Coffee Consumption

![Cloud Cover and Coffee Relationship](final_analysis_output/bulut_kahve_iliskisi.png)

ANOVA test was applied to test the difference in coffee consumption across cloud cover categories:
- F-value = 0.538, p = 0.658
- Effect size (eta-squared) = 0.021

Since p > 0.05, the null hypothesis cannot be rejected. There is no statistically significant difference in coffee consumption across cloud cover categories.

### Academic Events and Coffee Consumption

![Academic Events and Coffee Relationship](final_analysis_output/akademik_kahve_iliskisi.png)

Independent samples t-test was applied to test whether there is a difference in coffee consumption between exam/assignment days and normal days:
- t-value = -0.524, p = 0.602
- Effect size (Cohen's d) = -0.192
- 95% confidence interval: [-1.02, 0.60]

Since p > 0.05, the null hypothesis cannot be rejected. There is no statistically significant difference in coffee consumption between exam/assignment days and normal days.

### Sleep and Coffee Consumption

#### Sleep Duration and Coffee Consumption

![Sleep Duration and Coffee Relationship](final_analysis_output/uyku_kahve_iliskisi.png)

Pearson correlation test was applied to test the relationship between sleep duration and coffee consumption:
- r = -0.73, p < 0.001
- 95% confidence interval: [-0.82, -0.61]

These results show that coffee consumption significantly increases as sleep duration decreases. Since p < 0.001, the null hypothesis is strongly rejected, and there is a highly statistically significant relationship between sleep duration and coffee consumption.

#### Sleep Categories and Coffee Consumption

![Sleep Categories and Coffee Relationship](final_analysis_output/uyku_kategorisi_kahve_iliskisi.png)

ANOVA test was applied to test the difference in coffee consumption across sleep duration categories:
- F-value = 25.63, p < 0.001
- Effect size (eta-squared) = 0.413

Since p < 0.001, the null hypothesis is rejected. There is a highly statistically significant difference in coffee consumption across sleep duration categories. Post-hoc Tukey test shows that the insufficient sleep (<6 hours) category has statistically significantly higher coffee consumption than all other categories (p < 0.01).

#### Previous Day's Sleep Duration and Today's Coffee Consumption

![Previous Day's Sleep and Coffee Relationship](final_analysis_output/onceki_gun_uyku_kahve_iliskisi.png)

Pearson correlation test was applied to test the relationship between the previous day's sleep duration and today's coffee consumption:
- r = -0.15, p = 0.152
- 95% confidence interval: [-0.34, 0.05]

Since p > 0.05, the null hypothesis cannot be rejected. There is no statistically significant relationship between the previous day's sleep duration and today's coffee consumption.

### Combined Analysis of All Factors

#### Time Series Analysis

![Time Series of All Variables](final_analysis_output/tum_degiskenler_zaman_serisi.png)

Time series analysis shows the change of all variables over time. The exam/assignment days marked with red dashed lines in the graph represent the academic stress periods mentioned in our hypothesis. When examining the graph, the inverse relationship between sleep duration and coffee consumption is clearly visible.

#### Multiple Regression Analysis

Multiple regression analysis was applied to determine the factors affecting coffee consumption:

- Model statistics: F(5, 86) = 23.15, p < 0.001, Adjusted R² = 0.551
- Sleep duration: β = -1.0707, p < 0.001
- Temperature: β = -0.0148, p = 0.033
- Precipitation: β = 0.0075, p = 0.781
- Cloud cover: β = 0.0013, p = 0.325
- Academic event day: β = -0.1253, p = 0.542

These results show that sleep duration is the strongest factor affecting coffee consumption, and this relationship is highly statistically significant. Temperature also has a weak but statistically significant effect on coffee consumption.

![Two Most Influential Factors](final_analysis_output/en_etkili_faktorler_3d.png)

This three-dimensional graph shows the effect of the two most influential factors (sleep duration and weekday/weekend) on coffee consumption according to multiple regression analysis.

## Hypothesis Test Results

### Null Hypothesis

There is no significant relationship between coffee consumption and weather conditions, academic events, and sleep duration. Changes in daily coffee consumption are completely random and not affected by these external factors.

### Alternative Hypothesis

There is a significant relationship between coffee consumption and weather conditions, academic events, and sleep duration. Specifically, there is a correlation between lower temperatures, rainy/cloudy weather, stressful periods (exam weeks), and less sleep with higher coffee consumption.

### Conclusion

As a result of the statistical analyses, our alternative hypothesis has been partially confirmed:

1. **Sleep and Coffee Consumption**: 
   - Statistical test result: **Null hypothesis is rejected**. 
   - Strong negative correlation was found between sleep duration and coffee consumption (r = -0.73, p < 0.001).
   - It emerged as the most influential factor in regression analysis (β = -1.0707, p < 0.001).
   - Statistically significant difference was found in coffee consumption across sleep duration categories (F = 25.63, p < 0.001).

2. **Weather Conditions and Coffee Consumption**:
   - For temperature, the **null hypothesis is rejected** but the effect is weak.
   - Weak negative correlation was found between temperature and coffee consumption (r = -0.21, p = 0.043).
   - The weak but significant effect of temperature was confirmed in regression analysis (β = -0.0148, p = 0.033).
   - For precipitation and cloud cover variables, the **null hypothesis cannot be rejected** (p > 0.05).

3. **Academic Events and Coffee Consumption**:
   - Statistical test result: **Null hypothesis cannot be rejected**. 
   - No statistically significant difference was found in coffee consumption between exam/assignment days and normal days (t = -0.524, p = 0.602).
   - Contrary to expectations, lower coffee consumption was observed on exam/assignment days, although this difference is not statistically significant.

In conclusion, statistical analyses have determined that sleep duration is the most decisive factor affecting coffee consumption, only temperature from weather conditions has a limited effect, and academic stress factors did not show the expected effect.

## Future Work

In future studies, it could be beneficial to expand the analyses with more detailed measurements of academic stress factors (such as stress level surveys), longer data collection periods, and a larger participant pool. Additionally, using stronger statistical tests and larger sample sizes for each hypothesis could increase the reliability of the findings.