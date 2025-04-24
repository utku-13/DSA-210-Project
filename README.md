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

# Sonuçlar ve Veri Analizi

## Bulgular

Projenin temel hipotezlerini test etmek için çeşitli analizler gerçekleştirildi. Bu bölümde elde edilen bulgular ve hipotezlerin sonuçları paylaşılmıştır.

### Değişkenler Arası Korelasyon

![Korelasyon Matrisi](final_analysis_output/korelasyon_matrisi.png)

Korelasyon matrisi, kahve tüketimi ile diğer değişkenler arasındaki ilişkinin gücünü göstermektedir. Matristeki en güçlü ilişki, kahve tüketimi ve uyku süresi arasında negatif yönde bulunmaktadır (-0.73). Bu sonuç, hipotezimizde öngördüğümüz "daha az uyku, daha fazla kahve tüketimi" ilişkisini güçlü bir şekilde desteklemektedir. Hava durumu değişkenleri ile kahve tüketimi arasında ise daha zayıf ilişkiler gözlemlenmiştir.

### Hava Durumu ve Kahve Tüketimi

#### Sıcaklık ve Kahve Tüketimi

![Sıcaklık ve Kahve İlişkisi](final_analysis_output/sicaklik_kahve_iliskisi.png)

Sıcaklık ve kahve tüketimi arasındaki ilişkiyi gösteren bu grafik, hipotezimizde öne sürdüğümüz "düşük sıcaklık, yüksek kahve tüketimi" ilişkisini kısmen desteklemektedir. Korelasyon değeri -0.21 olarak hesaplanmıştır, bu da sıcaklık düştükçe kahve tüketiminin hafif bir artış gösterdiğini ancak bu ilişkinin çok güçlü olmadığını göstermektedir.

#### Yağış ve Kahve Tüketimi

![Yağış ve Kahve İlişkisi](final_analysis_output/yagis_kahve_iliskisi.png)

Yağış miktarı kategorilerine göre kahve tüketimini gösteren bu kutu grafiği, hipotezimizde belirtilen "yağışlı günlerde daha fazla kahve tüketimi" iddiasını net bir şekilde desteklememektedir. Grafikte yağış kategorileri arasında kahve tüketimi açısından belirgin bir fark gözlemlenmemektedir.

#### Bulut Örtüsü ve Kahve Tüketimi

![Bulut Örtüsü ve Kahve İlişkisi](final_analysis_output/bulut_kahve_iliskisi.png)

Bulut örtüsü kategorilerine göre kahve tüketimini gösteren bu kutu grafiği, hipotezimizde belirtilen "bulutlu/kapalı günlerde daha fazla kahve tüketimi" iddiasını desteklememektedir. Farklı bulut örtüsü kategorileri arasında kahve tüketimi açısından anlamlı bir fark gözlemlenmemektedir.

### Akademik Olaylar ve Kahve Tüketimi

![Akademik Olaylar ve Kahve İlişkisi](final_analysis_output/akademik_kahve_iliskisi.png)

Sınav/ödev teslim günleri ile normal günlerdeki kahve tüketimini karşılaştıran bu grafik, hipotezimizde öne sürdüğümüz "stresli akademik dönemlerde daha fazla kahve tüketimi" iddiasına ters düşmektedir. Grafiğe göre, sınav/ödev günlerinde ortalama kahve tüketimi (1.40 fincan) normal günlere (1.61 fincan) göre daha düşüktür. Ancak, istatistiksel analiz sonuçlarına göre (p-değeri: 0.6021) bu fark istatistiksel olarak anlamlı değildir.

### Uyku ve Kahve Tüketimi

#### Uyku Süresi ve Kahve Tüketimi

![Uyku Süresi ve Kahve İlişkisi](final_analysis_output/uyku_kahve_iliskisi.png)

Uyku süresi ve kahve tüketimi arasındaki ilişkiyi gösteren bu grafik, hipotezimizde öne sürdüğümüz "daha az uyku, daha fazla kahve tüketimi" iddiasını güçlü bir şekilde desteklemektedir. -0.73'lük korelasyon değeri, uyku süresi azaldıkça kahve tüketiminin belirgin bir şekilde arttığını göstermektedir. Bu, tüm değişkenler arasında kahve tüketimini en çok etkileyen faktör olarak öne çıkmaktadır.

#### Uyku Kategorileri ve Kahve Tüketimi

![Uyku Kategorileri ve Kahve İlişkisi](final_analysis_output/uyku_kategorisi_kahve_iliskisi.png)

Uyku süresi kategorilerine göre kahve tüketimini gösteren bu kutu grafiği, hipotezimizi desteklemektedir. Yetersiz uyku (<6 saat) kategorisinde kahve tüketimi belirgin bir şekilde daha yüksektir ve uyku süresi arttıkça kahve tüketimi düşmektedir.

#### Önceki Gün Uyku Süresi ve Bugünkü Kahve Tüketimi

![Önceki Gün Uyku ve Kahve İlişkisi](final_analysis_output/onceki_gun_uyku_kahve_iliskisi.png)

Önceki günün uyku süresi ile bugünkü kahve tüketimi arasındaki ilişkiyi gösteren bu grafik, hipotezimizin bir diğer yönünü desteklemektedir. Korelasyon değeri yüksek çıkmasa da (-0.15), genel eğilim, bir önceki gece daha az uyuyan kişinin ertesi gün daha fazla kahve tüketme eğiliminde olduğunu göstermektedir.

### Tüm Faktörlerin Birlikte Analizi

#### Zaman Serisi Analizi

![Tüm Değişkenler Zaman Serisi](final_analysis_output/tum_degiskenler_zaman_serisi.png)

Bu zaman serisi grafiği, tüm değişkenlerin zamanla nasıl değiştiğini göstermektedir. Grafikte kırmızı kesikli çizgilerle işaretlenen sınav/ödev günleri, hipotezimizde belirttiğimiz akademik stres dönemlerini temsil etmektedir. Grafik incelendiğinde, uyku süresi ve kahve tüketimi arasındaki ters ilişki açıkça görülmektedir.

#### En Etkili Faktörler

![En Etkili İki Faktör](final_analysis_output/en_etkili_faktorler_3d.png)

Çoklu regresyon analizi sonucunda en etkili iki faktörün (uyku süresi ve hafta içi/sonu) kahve tüketimi üzerindeki etkisini üç boyutlu gösteren bu grafik, hipotezlerimizin bir kısmını desteklemektedir. Uyku süresinin kahve tüketimi üzerindeki baskın etkisi (-1.0707) açıkça görülmektedir.

## Hipotez Testi Sonuçları

### Null Hipotezi

Kahve tüketimi ile hava koşulları, akademik olaylar ve uyku süresi arasında anlamlı bir ilişki yoktur. Günlük kahve tüketimindeki değişiklikler tamamen rastgeledir ve bu dış faktörlerden etkilenmemektedir.

### Alternatif Hipotez

Kahve tüketimi ile hava koşulları, akademik olaylar ve uyku süresi arasında anlamlı bir ilişki vardır. Özellikle düşük sıcaklıklar, yağışlı/bulutlu hava, stresli dönemler (sınav haftaları) ve daha az uyku ile daha yüksek kahve tüketimi arasında korelasyon vardır.

### Sonuç

Yapılan analizler sonucunda alternatif hipotezimiz kısmen doğrulanmıştır:

1. **Uyku ve Kahve Tüketimi**: 
   - Hipotez **güçlü bir şekilde doğrulandı**. 
   - Uyku süresi ve kahve tüketimi arasında güçlü bir negatif korelasyon (-0.73) bulundu.
   - Regresyon analizinde en etkili faktör olarak öne çıktı (-1.0707 katsayısı).

2. **Hava Koşulları ve Kahve Tüketimi**:
   - Hipotez **kısmen doğrulandı**.
   - Sıcaklık ile kahve tüketimi arasında zayıf negatif korelasyon (-0.21) bulundu.
   - Yağış ve bulut örtüsü değişkenleri için hipotez doğrulanmadı.

3. **Akademik Olaylar ve Kahve Tüketimi**:
   - Hipotez **doğrulanmadı**.
   - Sınav/ödev günlerinde beklenenin aksine daha düşük kahve tüketimi gözlemlendi.
   - Bu fark istatistiksel olarak anlamlı değildi (p=0.6021).

Sonuç olarak, kahve tüketimi üzerinde en belirleyici faktörün uyku süresi olduğu, hava koşullarının sınırlı bir etkiye sahip olduğu ve akademik stres faktörlerinin beklenen etkiyi göstermediği tespit edilmiştir.

## İleriki Çalışmalar

Gelecekteki çalışmalarda, akademik stres faktörlerinin daha ayrıntılı ölçümleri (örneğin stres seviyesi anketleri), daha uzun süreli veri toplama ve daha geniş katılımcı havuzu ile analizlerin genişletilmesi faydalı olabilir. 