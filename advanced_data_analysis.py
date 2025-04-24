import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# Stil ayarları
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# Veri dosyasını oku
print("Veri yükleniyor...")
df = pd.read_csv('raw_datas/coffee_sleep_data.csv')

# Tarihi datetime formatına çevir
df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y')

# ----- VERİ DÖNÜŞÜMÜ VE ZENGİNLEŞTİRME -----
print("\n1. Veri Dönüşümü ve Zenginleştirme Adımları:")

# Uyku saatlerini dakikaya çevir (analiz için)
df['SleepMinutes'] = df['SleepingHours'].apply(lambda x: int(x.split(':')[0])*60 + int(x.split(':')[1]))

# Uyku süresini saat olarak yeni bir sütuna ekle
df['SleepHours'] = df['SleepMinutes'] / 60

# Haftanın günlerini ekle
df['DayOfWeek'] = df['Date'].dt.day_name()
df['WeekDay'] = df['Date'].dt.dayofweek < 5  # Hafta içi: True, Hafta sonu: False
df['WeekDayStr'] = df['WeekDay'].map({True: 'Hafta İçi', False: 'Hafta Sonu'})
print("- Tarih bilgisinden haftanın günü ve hafta içi/sonu bilgisi türetildi")

# Ayları ekle
df['Month'] = df['Date'].dt.month_name()
print("- Tarih bilgisinden ay bilgisi türetildi")

# Kahve tüketimi kategorilerini ekle
bins = [-1, 0, 2, 5]  # -1 eklendi ki 0 değeri ilk kategoriye girsin
labels = ['Tüketim Yok', 'Az Tüketim', 'Yüksek Tüketim']
df['CoffeeCategory'] = pd.cut(df['CupsOfCoffee'], bins=bins, labels=labels)
print("- Kahve tüketimi kategorilere ayrıldı:", labels)

# Uyku kategorilerini ekle
sleep_bins = [0, 6, 7, 10]
sleep_labels = ['Yetersiz Uyku', 'Normal Uyku', 'Fazla Uyku']
df['SleepCategory'] = pd.cut(df['SleepHours'], bins=sleep_bins, labels=sleep_labels)
print("- Uyku süresi kategorilere ayrıldı:", sleep_labels)

# Z-score normalizasyonu
scaler = StandardScaler()
df['SleepHours_Norm'] = scaler.fit_transform(df[['SleepHours']])
df['CupsOfCoffee_Norm'] = scaler.fit_transform(df[['CupsOfCoffee']])
print("- Uyku ve kahve verileri standartlaştırıldı (z-score)")

# 1 günlük gecikme (lag) değerlerini hesapla
df['PrevDaySleep'] = df['SleepHours'].shift(1)
df['PrevDayCoffee'] = df['CupsOfCoffee'].shift(1)
print("- Önceki günün uyku ve kahve değerleri eklendi")

# Hareketli ortalamalar (3 gün)
df['CoffeeMA3'] = df['CupsOfCoffee'].rolling(window=3).mean()
df['SleepMA3'] = df['SleepHours'].rolling(window=3).mean()
print("- 3 günlük hareketli ortalamalar eklendi")

print(f"\nZenginleştirilmiş veri çerçevesi: {df.shape[0]} satır, {df.shape[1]} sütun")
print("Sütunlar:", df.columns.tolist())

# ----- KEŞİFSEL VERİ ANALİZİ (EDA) -----
print("\n2. Keşifsel Veri Analizi (EDA) Yapılıyor...")

# Çıktı klasörü oluştur
import os
if not os.path.exists('analysis_output'):
    os.makedirs('analysis_output')
    print("'analysis_output' klasörü oluşturuldu")

# Betimsel istatistikler
print("\nKahve Tüketimi İstatistikleri:")
print(df['CupsOfCoffee'].describe())

print("\nUyku Süresi İstatistikleri (saat):")
print(df['SleepHours'].describe())

# Korelasyon analizi
numeric_cols = ['CupsOfCoffee', 'SleepHours', 'PrevDaySleep', 'CoffeeMA3', 'SleepMA3']
correlation_matrix = df[numeric_cols].corr()
print("\nKorelasyon Matrisi:")
print(correlation_matrix)

# Kahve - Uyku korelasyonu
coffee_sleep_corr = df['CupsOfCoffee'].corr(df['SleepHours'])
print(f"\nKahve tüketimi ve uyku süresi arasındaki korelasyon: {coffee_sleep_corr:.3f}")

# Önceki günün uykusu - Kahve korelasyonu
prev_sleep_coffee_corr = df['CupsOfCoffee'].corr(df['PrevDaySleep'])
print(f"Bir önceki günün uyku süresi ve bugünkü kahve tüketimi arasındaki korelasyon: {prev_sleep_coffee_corr:.3f}")

# ------ GÖRSELLEŞTIRMELER ------

# 1. Korelasyon Matrisi Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Değişkenler Arası Korelasyon Matrisi')
plt.tight_layout()
plt.savefig('analysis_output/korelasyon_matrisi.png')

# 2. Kahve ve Uyku İlişkisi (gelişmiş scatter plot)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='SleepHours', y='CupsOfCoffee', hue='WeekDayStr', 
                size='CupsOfCoffee', sizes=(50, 200), 
                palette='viridis', data=df)
plt.xlabel('Uyku Süresi (Saat)')
plt.ylabel('Kahve Fincanı Sayısı')
plt.title('Kahve Tüketimi ve Uyku Süresi İlişkisi')
plt.grid(True, alpha=0.3)

# Regresyon çizgisi
sns.regplot(x='SleepHours', y='CupsOfCoffee', data=df, scatter=False, 
            line_kws={"color":"red", "alpha":0.7, "lw":2, "ls":"--"})

plt.legend(title='Gün Türü')
plt.tight_layout()
plt.savefig('analysis_output/kahve_uyku_iliskisi.png')

# 3. Zaman İçinde Kahve ve Uyku
plt.figure(figsize=(14, 7))
ax1 = plt.gca()
ax2 = ax1.twinx()

# Kahve çizgisi
ax1.plot(df['Date'], df['CupsOfCoffee'], marker='o', color='#1f77b4', linewidth=2, label='Kahve (Fincan)')
ax1.set_ylabel('Kahve Fincanı', color='#1f77b4')
ax1.tick_params(axis='y', colors='#1f77b4')
ax1.set_ylim(bottom=0)

# Uyku çizgisi
ax2.plot(df['Date'], df['SleepHours'], marker='s', color='#ff7f0e', linewidth=2, label='Uyku (Saat)')
ax2.set_ylabel('Uyku Süresi (Saat)', color='#ff7f0e')
ax2.tick_params(axis='y', colors='#ff7f0e')

# Haftasonlarını işaretle
weekend_dates = df[df['WeekDay'] == False]['Date']
for date in weekend_dates:
    plt.axvline(x=date, color='lightgray', linestyle='--', alpha=0.5)

# Genel ayarlar
plt.title('Zaman İçinde Kahve Tüketimi ve Uyku Süresi')
plt.xlabel('Tarih')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# Özel bir legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.tight_layout()
plt.savefig('analysis_output/zaman_serisi_analizi.png')

# 4. Hafta İçi vs Hafta Sonu Karşılaştırması
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.boxplot(x='WeekDayStr', y='CupsOfCoffee', data=df)
plt.title('Hafta İçi vs Hafta Sonu Kahve Tüketimi')
plt.xlabel('')
plt.ylabel('Kahve Fincanı Sayısı')

plt.subplot(1, 2, 2)
sns.boxplot(x='WeekDayStr', y='SleepHours', data=df)
plt.title('Hafta İçi vs Hafta Sonu Uyku Süresi')
plt.xlabel('')
plt.ylabel('Uyku Süresi (Saat)')

plt.tight_layout()
plt.savefig('analysis_output/hafta_ici_hafta_sonu.png')

# 5. Kahve tüketimi ve uyku kategorileri
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.countplot(x='CoffeeCategory', data=df, palette='Blues_r')
plt.title('Kahve Tüketimi Kategorileri')
plt.xlabel('')
plt.ylabel('Gün Sayısı')

plt.subplot(1, 2, 2)
sns.countplot(x='SleepCategory', data=df, palette='YlOrBr')
plt.title('Uyku Süresi Kategorileri')
plt.xlabel('')
plt.ylabel('Gün Sayısı')

plt.tight_layout()
plt.savefig('analysis_output/kategori_dagilimi.png')

# 6. Gün bazında ortalama kahve tüketimi
plt.figure(figsize=(10, 6))
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_coffee = df.groupby('DayOfWeek')['CupsOfCoffee'].mean().reindex(day_order)
day_coffee.plot(kind='bar', color='skyblue')
plt.axhline(y=df['CupsOfCoffee'].mean(), color='red', linestyle='--', 
            label=f'Genel Ortalama: {df["CupsOfCoffee"].mean():.2f}')
plt.title('Haftanın Günlerine Göre Ortalama Kahve Tüketimi')
plt.xlabel('Gün')
plt.ylabel('Ortalama Kahve Fincanı')
plt.xticks(rotation=45)
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('analysis_output/gunlere_gore_kahve.png')

# 7. Uyku - Kahve dağılımı (kategorilere göre)
plt.figure(figsize=(12, 8))
sns.boxplot(x='SleepCategory', y='CupsOfCoffee', data=df, palette='YlOrBr')
plt.title('Uyku Kategorilerine Göre Kahve Tüketimi')
plt.xlabel('Uyku Kategorisi')
plt.ylabel('Kahve Fincanı Sayısı')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('analysis_output/uyku_kahve_dagilimi.png')

# ----- HİPOTEZ TESTLERİ -----
print("\n3. Hipotez Testleri Yapılıyor...")

print("\nHipotez 1: Uyku süresi ile kahve tüketimi arasında negatif bir korelasyon vardır.")
print("H0: Uyku süresi ve kahve tüketimi arasında korelasyon yoktur (ρ = 0)")
print("H1: Uyku süresi ve kahve tüketimi arasında negatif korelasyon vardır (ρ < 0)")

corr, p_value = stats.pearsonr(df['SleepHours'], df['CupsOfCoffee'])
print(f"Korelasyon katsayısı (r): {corr:.3f}")
print(f"P-değeri: {p_value/2:.4f} (Tek yönlü test için p/2)")  # Tek yönlü test için p/2

alpha = 0.05
if p_value/2 < alpha and corr < 0:
    print(f"Sonuç: p-değeri ({p_value/2:.4f}) < alfa ({alpha}), negatif korelasyon.")
    print("H0 hipotezi reddedilir. Uyku süresi ve kahve tüketimi arasında istatistiksel olarak anlamlı bir negatif korelasyon vardır.")
elif p_value/2 < alpha and corr > 0:
    print(f"Sonuç: p-değeri ({p_value/2:.4f}) < alfa ({alpha}), ancak korelasyon pozitif.")
    print("H0 hipotezi reddedilir, ancak ilişki beklenenin tersine pozitiftir.")
else:
    print(f"Sonuç: p-değeri ({p_value/2:.4f}) > alfa ({alpha}).")
    print("H0 hipotezi reddedilemez. Uyku süresi ve kahve tüketimi arasında istatistiksel olarak anlamlı bir negatif korelasyon yoktur.")

print("\nHipotez 2: Hafta içi ve hafta sonu günlerde kahve tüketimi farklıdır.")
print("H0: Hafta içi ve hafta sonu günlerde kahve tüketim ortalamaları arasında fark yoktur")
print("H1: Hafta içi ve hafta sonu günlerde kahve tüketim ortalamaları arasında fark vardır")

weekday_coffee = df[df['WeekDay'] == True]['CupsOfCoffee']
weekend_coffee = df[df['WeekDay'] == False]['CupsOfCoffee']

print(f"Hafta içi ortalama kahve tüketimi: {weekday_coffee.mean():.2f} fincan")
print(f"Hafta sonu ortalama kahve tüketimi: {weekend_coffee.mean():.2f} fincan")

t_stat, p_value = stats.ttest_ind(weekday_coffee, weekend_coffee, equal_var=False)
print(f"T-istatistiği: {t_stat:.3f}, P-değeri: {p_value:.4f}")

if p_value < alpha:
    print(f"Sonuç: p-değeri ({p_value:.4f}) < alfa ({alpha}).")
    print("H0 hipotezi reddedilir. Hafta içi ve hafta sonu kahve tüketiminde istatistiksel olarak anlamlı bir fark vardır.")
else:
    print(f"Sonuç: p-değeri ({p_value:.4f}) > alfa ({alpha}).")
    print("H0 hipotezi reddedilemez. Hafta içi ve hafta sonu kahve tüketiminde istatistiksel olarak anlamlı bir fark yoktur.")

print("\nHipotez 3: Yetersiz uyku günlerinde daha fazla kahve tüketilir.")
print("H0: Yetersiz uyku günleri ve normal/fazla uyku günleri arasında kahve tüketiminde fark yoktur")
print("H1: Yetersiz uyku günlerinde daha fazla kahve tüketilir")

insufficient_sleep = df[df['SleepCategory'] == 'Yetersiz Uyku']['CupsOfCoffee']
sufficient_sleep = df[df['SleepCategory'] != 'Yetersiz Uyku']['CupsOfCoffee']

print(f"Yetersiz uyku günlerinde ortalama kahve tüketimi: {insufficient_sleep.mean():.2f} fincan")
print(f"Normal/fazla uyku günlerinde ortalama kahve tüketimi: {sufficient_sleep.mean():.2f} fincan")

t_stat, p_value = stats.ttest_ind(insufficient_sleep, sufficient_sleep, equal_var=False, alternative='greater')
print(f"T-istatistiği: {t_stat:.3f}, P-değeri: {p_value:.4f} (Tek yönlü test)")

if p_value < alpha:
    print(f"Sonuç: p-değeri ({p_value:.4f}) < alfa ({alpha}).")
    print("H0 hipotezi reddedilir. Yetersiz uyku günlerinde istatistiksel olarak anlamlı şekilde daha fazla kahve tüketilir.")
else:
    print(f"Sonuç: p-değeri ({p_value:.4f}) > alfa ({alpha}).")
    print("H0 hipotezi reddedilemez. Yetersiz uyku günlerinde istatistiksel olarak anlamlı şekilde daha fazla kahve tüketildiği söylenemez.")

print("\nHipotez 4: Önceki günün uyku süresi, bugünkü kahve tüketimini etkiler.")
print("H0: Önceki günün uyku süresi ve bugünkü kahve tüketimi arasında korelasyon yoktur")
print("H1: Önceki günün uyku süresi ve bugünkü kahve tüketimi arasında negatif korelasyon vardır")

# NaN değerlerini kaldır
valid_data = df.dropna(subset=['PrevDaySleep', 'CupsOfCoffee'])
corr, p_value = stats.pearsonr(valid_data['PrevDaySleep'], valid_data['CupsOfCoffee'])
print(f"Korelasyon katsayısı (r): {corr:.3f}")
print(f"P-değeri: {p_value/2:.4f} (Tek yönlü test için p/2)")

if p_value/2 < alpha and corr < 0:
    print(f"Sonuç: p-değeri ({p_value/2:.4f}) < alfa ({alpha}), negatif korelasyon.")
    print("H0 hipotezi reddedilir. Önceki günün uyku süresi ve bugünkü kahve tüketimi arasında istatistiksel olarak anlamlı bir negatif korelasyon vardır.")
elif p_value/2 < alpha and corr > 0:
    print(f"Sonuç: p-değeri ({p_value/2:.4f}) < alfa ({alpha}), ancak korelasyon pozitif.")
    print("H0 hipotezi reddedilir, ancak ilişki beklenenin tersine pozitiftir.")
else:
    print(f"Sonuç: p-değeri ({p_value/2:.4f}) > alfa ({alpha}).")
    print("H0 hipotezi reddedilemez. Önceki günün uyku süresi ve bugünkü kahve tüketimi arasında istatistiksel olarak anlamlı bir negatif korelasyon yoktur.")

print("\nAnaliz tamamlandı. Tüm görseller 'analysis_output' klasörüne kaydedildi.") 