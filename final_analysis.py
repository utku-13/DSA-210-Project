import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# Stil ayarları
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# Çıktı klasörü oluştur
output_dir = 'final_analysis_output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"'{output_dir}' klasörü oluşturuldu")

def load_data():
    """Veri setlerini yükle ve birleştir"""
    print("Veri setleri yükleniyor...")
    
    # Kahve ve uyku verileri
    coffee_sleep_file = 'raw_datas/coffee_sleep_data.csv'
    if not os.path.exists(coffee_sleep_file):
        print(f"Hata: {coffee_sleep_file} dosyası bulunamadı!")
        return None
    
    # Hava durumu verileri
    weather_file = 'raw_datas/weather_data_istanbul.csv'
    if not os.path.exists(weather_file):
        print(f"Hata: {weather_file} dosyası bulunamadı!")
        return None
    
    # Akademik takvim verileri
    academic_file = 'raw_datas/academic_calendar.csv'
    if not os.path.exists(academic_file):
        print(f"Hata: {academic_file} dosyası bulunamadı!")
        return None
    
    # Verileri oku
    coffee_sleep_df = pd.read_csv(coffee_sleep_file)
    weather_df = pd.read_csv(weather_file)
    academic_df = pd.read_csv(academic_file)
    
    # Tarih sütunlarını doğru formata dönüştür
    coffee_sleep_df['Date'] = pd.to_datetime(coffee_sleep_df['Date'], format='%d.%m.%Y')
    weather_df['date'] = pd.to_datetime(weather_df['date'], format='%d.%m.%Y')
    academic_df['Date'] = pd.to_datetime(academic_df['Date'], format='%d.%m.%Y')
    
    # Boolean sütununu doğru formata dönüştür
    academic_df['DoYouHaveAnySubmissionOrExam'] = academic_df['DoYouHaveAnySubmissionOrExam'].map({'true': True, 'false': False})
    
    # Eğer string ise, convert metni küçük harfe çevir ve True/False ile karşılaştır
    if academic_df['DoYouHaveAnySubmissionOrExam'].dtype == 'object':
        academic_df['DoYouHaveAnySubmissionOrExam'] = academic_df['DoYouHaveAnySubmissionOrExam'].astype(str).str.lower().map({'true': True, 'false': False})
    
    # Birleştirme için sütun isimlerini eşleştir
    weather_df.rename(columns={'date': 'Date'}, inplace=True)
    
    # Verileri birleştir (inner join - sadece her iki sette de var olan tarihler)
    merged_df = pd.merge(coffee_sleep_df, weather_df, on='Date', how='inner')
    merged_df = pd.merge(merged_df, academic_df, on='Date', how='inner')
    
    # Uyku saatlerini dakikaya çevir (analiz için)
    merged_df['SleepMinutes'] = merged_df['SleepingHours'].apply(lambda x: int(x.split(':')[0])*60 + int(x.split(':')[1]))
    merged_df['SleepHours'] = merged_df['SleepMinutes'] / 60
    
    # Haftanın günlerini ekle
    merged_df['DayOfWeek'] = merged_df['Date'].dt.day_name()
    merged_df['WeekDay'] = merged_df['Date'].dt.dayofweek < 5  # Hafta içi: True, Hafta sonu: False
    merged_df['WeekDayStr'] = merged_df['WeekDay'].map({True: 'Hafta İçi', False: 'Hafta Sonu'})
    
    # Kategorileri Boolean'dan String'e çevir
    merged_df['HasExamOrSubmission'] = merged_df['DoYouHaveAnySubmissionOrExam'].map({True: 'Var', False: 'Yok'})
    
    print(f"Birleştirilmiş veri seti: {merged_df.shape[0]} satır, {merged_df.shape[1]} sütun")
    print("Boolean sütun durumu:", merged_df['DoYouHaveAnySubmissionOrExam'].value_counts())
    
    return merged_df

def analyze_correlations(df):
    """Değişkenler arası korelasyonları analiz et ve görselleştir"""
    print("\nDeğişkenler arası korelasyonlar analiz ediliyor...")
    
    # Ana değişkenler arasındaki korelasyonları hesapla
    numeric_cols = ['CupsOfCoffee', 'SleepHours', 'avg_temp', 'precipitation', 'cloud_cover']
    correlation_matrix = df[numeric_cols].corr()
    
    # Korelasyon matrisi heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Değişkenler Arası Korelasyon Matrisi')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/korelasyon_matrisi.png')
    
    # Önemli korelasyonları yazdır
    print("\nÖnemli Korelasyonlar:")
    for col in numeric_cols:
        if col != 'CupsOfCoffee':
            corr = correlation_matrix.loc['CupsOfCoffee', col]
            direction = "pozitif" if corr > 0 else "negatif"
            strength = "güçlü" if abs(corr) > 0.5 else "orta" if abs(corr) > 0.3 else "zayıf"
            print(f"Kahve tüketimi ve {col} arasında {strength} {direction} korelasyon: {corr:.3f}")

def weather_coffee_analysis(df):
    """Hava durumu ve kahve tüketimi ilişkisini analiz et"""
    print("\nHava durumu ve kahve tüketimi ilişkisi analiz ediliyor...")
    
    # 1. Sıcaklık ve kahve tüketimi
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x='avg_temp', y='CupsOfCoffee', data=df, alpha=0.7, s=100)
    
    # Regresyon çizgisi
    sns.regplot(x='avg_temp', y='CupsOfCoffee', data=df, scatter=False, 
                line_kws={"color":"red", "alpha":0.7, "lw":2, "ls":"--"})
    
    plt.xlabel('Ortalama Sıcaklık (°C)')
    plt.ylabel('Kahve Fincanı Sayısı')
    plt.title('Sıcaklık ve Kahve Tüketimi İlişkisi')
    
    # Korelasyon değerini ekle
    corr = df['avg_temp'].corr(df['CupsOfCoffee'])
    plt.annotate(f"Korelasyon: {corr:.2f}", xy=(0.05, 0.95), xycoords='axes fraction', 
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/sicaklik_kahve_iliskisi.png')
    
    # 2. Yağış ve kahve tüketimi
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=pd.cut(df['precipitation'], bins=[0, 0.1, 1, 2, 100], 
                         labels=['Yağış Yok', 'Az Yağış', 'Orta Yağış', 'Çok Yağış']), 
                y='CupsOfCoffee', data=df)
    
    plt.xlabel('Yağış Kategorisi')
    plt.ylabel('Kahve Fincanı Sayısı')
    plt.title('Yağış Miktarı ve Kahve Tüketimi İlişkisi')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/yagis_kahve_iliskisi.png')
    
    # 3. Bulut örtüsü ve kahve tüketimi
    plt.figure(figsize=(12, 6))
    
    # Bulut örtüsü kategorileri oluştur
    df['cloud_category'] = pd.cut(df['cloud_cover'], 
                                 bins=[0, 25, 50, 75, 100], 
                                 labels=['Az Bulutlu (0-25%)', 'Parçalı Bulutlu (25-50%)', 
                                         'Çok Bulutlu (50-75%)', 'Kapalı (75-100%)'])
    
    sns.boxplot(x='cloud_category', y='CupsOfCoffee', data=df)
    plt.xlabel('Bulut Örtüsü Kategorisi')
    plt.ylabel('Kahve Fincanı Sayısı')
    plt.title('Bulut Örtüsü ve Kahve Tüketimi İlişkisi')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/bulut_kahve_iliskisi.png')

def academic_coffee_analysis(df):
    """Akademik etkinlikler ve kahve tüketimi ilişkisini analiz et"""
    print("\nAkademik etkinlikler ve kahve tüketimi ilişkisi analiz ediliyor...")
    
    # Sınav/ödev teslimi günlerini kontrol et
    exam_days = df[df['DoYouHaveAnySubmissionOrExam'] == True]
    no_exam_days = df[df['DoYouHaveAnySubmissionOrExam'] == False]
    
    print(f"\nSınav/Ödev Günleri vs Normal Günler Analizi:")
    print(f"Sınav/Ödev günü sayısı: {len(exam_days)}")
    print(f"Normal gün sayısı: {len(no_exam_days)}")
    
    if len(exam_days) > 0 and len(no_exam_days) > 0:
        # Sınav/ödev teslimi günleri ve kahve tüketimi karşılaştırması
        plt.figure(figsize=(10, 6))
        
        # Barplot için veri hazırlığı
        exam_summary = pd.DataFrame({
            'Durum': ['Sınav/Ödev Günleri', 'Normal Günler'],
            'Ortalama Kahve': [exam_days['CupsOfCoffee'].mean(), no_exam_days['CupsOfCoffee'].mean()]
        })
        
        sns.barplot(x='Durum', y='Ortalama Kahve', data=exam_summary)
        
        plt.xlabel('Sınav veya Ödev Teslimi Durumu')
        plt.ylabel('Ortalama Kahve Fincanı Sayısı')
        plt.title('Sınav/Ödev Teslimi Günlerinde Kahve Tüketimi')
        
        # Ortalama değerleri yazdır
        for i, val in enumerate(exam_summary['Ortalama Kahve']):
            plt.text(i, val + 0.1, f'{val:.2f}', ha='center')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/akademik_kahve_iliskisi.png')
        
        # Detaylı analiz için t-testi yapalım
        import scipy.stats as stats
        
        exam_coffee = exam_days['CupsOfCoffee']
        no_exam_coffee = no_exam_days['CupsOfCoffee']
        
        if len(exam_coffee) >= 2 and len(no_exam_coffee) >= 2:
            t_stat, p_value = stats.ttest_ind(exam_coffee, no_exam_coffee, equal_var=False)
            
            print(f"Sınav/Ödev günlerinde ortalama kahve tüketimi: {exam_coffee.mean():.2f} fincan")
            print(f"Normal günlerde ortalama kahve tüketimi: {no_exam_coffee.mean():.2f} fincan")
            print(f"T-istatistiği: {t_stat:.3f}, P-değeri: {p_value:.4f}")
            
            # Anlamlılık testini yazdır
            alpha = 0.05
            if p_value < alpha:
                print(f"Sonuç: İstatistiksel olarak anlamlı bir fark bulundu (p<{alpha}).")
                if exam_coffee.mean() > no_exam_coffee.mean():
                    print("Sınav/Ödev günlerinde kahve tüketimi anlamlı bir şekilde daha yüksek.")
                else:
                    print("Sınav/Ödev günlerinde kahve tüketimi anlamlı bir şekilde daha düşük.")
            else:
                print(f"Sonuç: İstatistiksel olarak anlamlı bir fark bulunamadı (p>{alpha}).")
        else:
            print("Yeterli veri yok: T-testi uygulanamadı (her grupta en az 2 veri noktası gerekli).")
            print(f"Sınav/Ödev günlerinde ortalama kahve tüketimi: {exam_coffee.mean():.2f} fincan")
            print(f"Normal günlerde ortalama kahve tüketimi: {no_exam_coffee.mean():.2f} fincan")
    else:
        print("Sınav/ödev günleri veya normal günler için yeterli veri yok. Analiz yapılamadı.")

def sleep_coffee_analysis(df):
    """Uyku ve kahve tüketimi ilişkisini analiz et"""
    print("\nUyku ve kahve tüketimi ilişkisi analiz ediliyor...")
    
    # 1. Uyku süresi ve kahve tüketimi arasındaki ilişki
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x='SleepHours', y='CupsOfCoffee', hue='WeekDayStr', 
                    size='CupsOfCoffee', sizes=(50, 200), 
                    palette='viridis', data=df)
    
    # Regresyon çizgisi
    sns.regplot(x='SleepHours', y='CupsOfCoffee', data=df, scatter=False, 
                line_kws={"color":"red", "alpha":0.7, "lw":2, "ls":"--"})
    
    plt.xlabel('Uyku Süresi (Saat)')
    plt.ylabel('Kahve Fincanı Sayısı')
    plt.title('Uyku Süresi ve Kahve Tüketimi İlişkisi')
    
    # Korelasyon değerini ekle
    corr = df['SleepHours'].corr(df['CupsOfCoffee'])
    plt.annotate(f"Korelasyon: {corr:.2f}", xy=(0.05, 0.95), xycoords='axes fraction', 
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.grid(True, alpha=0.3)
    plt.legend(title='Gün Türü')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/uyku_kahve_iliskisi.png')
    
    # 2. Uyku kategorilerine göre kahve tüketimi
    # Uyku kategorileri oluştur
    df['sleep_category'] = pd.cut(df['SleepHours'], 
                                 bins=[0, 6, 7, 10], 
                                 labels=['Yetersiz Uyku (<6 saat)', 'Normal Uyku (6-7 saat)', 
                                         'Uzun Uyku (>7 saat)'])
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='sleep_category', y='CupsOfCoffee', data=df)
    plt.xlabel('Uyku Süresi Kategorisi')
    plt.ylabel('Kahve Fincanı Sayısı')
    plt.title('Uyku Süresi Kategorilerine Göre Kahve Tüketimi')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/uyku_kategorisi_kahve_iliskisi.png')
    
    # 3. Önceki gün uyku süresi ve bugünkü kahve tüketimi
    # Önceki günün uyku süresini hesapla
    df['PrevDaySleep'] = df['SleepHours'].shift(1)
    
    # NaN değerlerini kaldır
    prev_sleep_df = df.dropna(subset=['PrevDaySleep'])
    
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x='PrevDaySleep', y='CupsOfCoffee', hue='WeekDayStr', 
                    size='CupsOfCoffee', sizes=(50, 200), 
                    palette='viridis', data=prev_sleep_df)
    
    # Regresyon çizgisi
    sns.regplot(x='PrevDaySleep', y='CupsOfCoffee', data=prev_sleep_df, scatter=False, 
                line_kws={"color":"red", "alpha":0.7, "lw":2, "ls":"--"})
    
    plt.xlabel('Önceki Günün Uyku Süresi (Saat)')
    plt.ylabel('Kahve Fincanı Sayısı')
    plt.title('Önceki Günün Uyku Süresi ve Bugünkü Kahve Tüketimi İlişkisi')
    
    # Korelasyon değerini ekle
    corr = prev_sleep_df['PrevDaySleep'].corr(prev_sleep_df['CupsOfCoffee'])
    plt.annotate(f"Korelasyon: {corr:.2f}", xy=(0.05, 0.95), xycoords='axes fraction', 
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.grid(True, alpha=0.3)
    plt.legend(title='Gün Türü')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/onceki_gun_uyku_kahve_iliskisi.png')

def combined_factors_analysis(df):
    """Tüm faktörlerin birlikte etkisini analiz et"""
    print("\nTüm faktörlerin birlikte etkisi analiz ediliyor...")
    
    # 1. Zaman serisi grafiği - Tüm değişkenleri birlikte göster
    plt.figure(figsize=(14, 10))
    
    # Birinci grafik: Kahve tüketimi
    ax1 = plt.subplot(4, 1, 1)
    ax1.plot(df['Date'], df['CupsOfCoffee'], 'o-', color='brown', markersize=6, linewidth=2)
    ax1.set_ylabel('Kahve (Fincan)')
    ax1.set_title('Kahve Tüketimi Zaman Serisi')
    ax1.grid(True, alpha=0.3)
    
    # Sınavların olduğu günleri işaretle
    exam_days = df[df['DoYouHaveAnySubmissionOrExam'] == True]
    if not exam_days.empty:
        for date in exam_days['Date']:
            ax1.axvline(x=date, color='red', linestyle='--', alpha=0.3)
    
    # İkinci grafik: Uyku süresi
    ax2 = plt.subplot(4, 1, 2, sharex=ax1)
    ax2.plot(df['Date'], df['SleepHours'], 'o-', color='blue', markersize=6, linewidth=2)
    ax2.set_ylabel('Uyku (Saat)')
    ax2.set_title('Uyku Süresi Zaman Serisi')
    ax2.grid(True, alpha=0.3)
    
    # Üçüncü grafik: Sıcaklık
    ax3 = plt.subplot(4, 1, 3, sharex=ax1)
    ax3.plot(df['Date'], df['avg_temp'], 'o-', color='red', markersize=6, linewidth=2)
    ax3.set_ylabel('Sıcaklık (°C)')
    ax3.set_title('Ortalama Sıcaklık Zaman Serisi')
    ax3.grid(True, alpha=0.3)
    
    # Dördüncü grafik: Yağış ve bulut örtüsü
    ax4 = plt.subplot(4, 1, 4, sharex=ax1)
    ax4.bar(df['Date'], df['precipitation'], color='skyblue', alpha=0.7, label='Yağış (mm)')
    
    ax5 = ax4.twinx()
    ax5.plot(df['Date'], df['cloud_cover'], 'o-', color='navy', markersize=4, linewidth=1.5, label='Bulut Örtüsü (%)')
    ax5.set_ylabel('Bulut Örtüsü (%)')
    
    ax4.set_ylabel('Yağış (mm)')
    ax4.set_title('Yağış ve Bulut Örtüsü Zaman Serisi')
    ax4.set_xlabel('Tarih')
    ax4.grid(True, alpha=0.3)
    
    # El ile legend oluştur
    lines4, labels4 = ax4.get_legend_handles_labels()
    lines5, labels5 = ax5.get_legend_handles_labels()
    ax4.legend(lines4 + lines5, labels4 + labels5, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/tum_degiskenler_zaman_serisi.png')
    
    # 2. Basit çoklu regresyon analizi (sklearn kullanarak)
    from sklearn.linear_model import LinearRegression
    
    print("\nÇoklu Regresyon Analizi:")
    
    # Veri hazırlığı - sayısal değişkenler
    X_numeric = df[['SleepHours', 'avg_temp', 'precipitation', 'cloud_cover']].values
    
    # Kategorik değişkenler - manuel dönüştürme
    has_exam = np.array([1 if x else 0 for x in df['DoYouHaveAnySubmissionOrExam']]).reshape(-1, 1)
    is_weekday = np.array([1 if x else 0 for x in df['WeekDay']]).reshape(-1, 1)
    
    # Tüm özellikleri birleştir
    X = np.concatenate([X_numeric, has_exam, is_weekday], axis=1)
    y = df['CupsOfCoffee'].values
    
    # Model oluştur ve eğit
    model = LinearRegression()
    model.fit(X, y)
    
    # Sonuçları görüntüle
    feature_names = ['Uyku Süresi', 'Sıcaklık', 'Yağış', 'Bulut Örtüsü', 'Sınav/Ödev', 'Hafta İçi']
    
    print("\nKatsayılar:")
    for name, coef in zip(feature_names, model.coef_):
        direction = "artırıcı" if coef > 0 else "azaltıcı"
        print(f"- {name}: Kahve tüketimi üzerinde {direction} etki (katsayı={coef:.4f})")
    
    print(f"\nSabit (Intercept): {model.intercept_:.4f}")
    print(f"Model R² skoru: {model.score(X, y):.4f}")
    
    # En etkili faktörleri belirle
    abs_coefs = np.abs(model.coef_)
    sorted_idx = np.argsort(abs_coefs)[::-1]  # Büyükten küçüğe sırala
    
    print("\nEn etkili faktörler (mutlak değer büyüklüğüne göre):")
    for i in range(len(sorted_idx)):
        idx = sorted_idx[i]
        print(f"{i+1}. {feature_names[idx]}: {model.coef_[idx]:.4f}")
    
    # En etkili iki faktör için 3D görselleştirme
    if len(sorted_idx) >= 2:
        top_idx1 = sorted_idx[0]
        top_idx2 = sorted_idx[1]
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        factor1 = X[:, top_idx1]
        factor2 = X[:, top_idx2]
        
        scatter = ax.scatter(factor1, factor2, y, c=y, cmap='viridis', s=50, alpha=0.8)
        
        ax.set_xlabel(feature_names[top_idx1])
        ax.set_ylabel(feature_names[top_idx2])
        ax.set_zlabel('Kahve Tüketimi (Fincan)')
        
        plt.colorbar(scatter, ax=ax, label='Kahve Tüketimi (Fincan)')
        plt.title(f'En Etkili İki Faktör: {feature_names[top_idx1]} ve {feature_names[top_idx2]}')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/en_etkili_faktorler_3d.png')

def main():
    """Ana fonksiyon: Tüm analizleri çalıştırır"""
    print("Kahve Tüketimi ve Dış Faktörler Analizi")
    print("========================================\n")
    
    # Verileri yükle ve birleştir
    df = load_data()
    
    if df is None or df.empty:
        print("Hata: Veri yüklenemedi veya birleştirilemedi.")
        return
    
    # Korelasyon analizi
    analyze_correlations(df)
    
    # Hava durumu ve kahve tüketimi ilişkisi
    weather_coffee_analysis(df)
    
    # Akademik etkinlikler ve kahve tüketimi ilişkisi
    academic_coffee_analysis(df)
    
    # Uyku ve kahve tüketimi ilişkisi
    sleep_coffee_analysis(df)
    
    # Tüm faktörlerin birlikte etkisi
    combined_factors_analysis(df)
    
    print(f"\nAnaliz tamamlandı. Tüm görseller '{output_dir}' klasörüne kaydedildi.")

if __name__ == "__main__":
    main() 