import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def merge_datasets():
    """
    Kahve tüketimi, uyku ve hava durumu verilerini tarih sütununa göre birleştirir.
    """
    print("Veri setleri birleştiriliyor...")
    
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
    
    # Verileri oku
    coffee_sleep_df = pd.read_csv(coffee_sleep_file)
    weather_df = pd.read_csv(weather_file)
    
    # Tarih sütunlarını doğru formata dönüştür
    coffee_sleep_df['Date'] = pd.to_datetime(coffee_sleep_df['Date'], format='%d.%m.%Y')
    weather_df['date'] = pd.to_datetime(weather_df['date'], format='%d.%m.%Y')
    
    # Birleştirme için sütun isimlerini eşleştir
    weather_df.rename(columns={'date': 'Date'}, inplace=True)
    
    # İki veri setini birleştir (inner join - sadece her iki sette de var olan tarihler)
    merged_df = pd.merge(coffee_sleep_df, weather_df, on='Date', how='inner')
    
    print(f"Birleştirilmiş veri seti: {merged_df.shape[0]} satır, {merged_df.shape[1]} sütun")
    
    # Eksik verileri kontrol et
    if merged_df.isnull().sum().sum() > 0:
        print("\nEksik veriler:")
        print(merged_df.isnull().sum())
        
        # Eksik verileri doldur (uygun stratejiye göre)
        print("Eksik veriler ortalama değerlerle dolduruluyor...")
        merged_df.fillna(merged_df.mean(numeric_only=True), inplace=True)
    
    # Birleştirilmiş veriyi kaydet
    output_file = 'raw_datas/coffee_sleep_weather_data.csv'
    merged_df.to_csv(output_file, index=False)
    print(f"Birleştirilmiş veri {output_file} dosyasına kaydedildi.")
    
    return merged_df

def create_visualizations(df):
    """
    Birleştirilmiş veriler için temel görselleştirmeler oluşturur.
    """
    if df is None or df.empty:
        print("Görselleştirme için geçerli veri yok!")
        return
    
    print("\nGörselleştirmeler oluşturuluyor...")
    
    # Çıktı klasörü oluştur
    output_dir = 'combined_analysis'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"'{output_dir}' klasörü oluşturuldu")
    
    # 1. Korelasyon matrisi
    plt.figure(figsize=(10, 8))
    numeric_cols = ['CupsOfCoffee', 'SleepingHours', 'avg_temp', 'precipitation', 'cloud_cover']
    
    # Uyku saatlerini dakikaya çevir
    if 'SleepingHours' in df.columns:
        df['SleepHours'] = df['SleepingHours'].apply(
            lambda x: int(str(x).split(':')[0]) + int(str(x).split(':')[1])/60 if ':' in str(x) else x
        )
        numeric_cols = ['CupsOfCoffee', 'SleepHours', 'avg_temp', 'precipitation', 'cloud_cover']
    
    # Korelasyon matrisi
    correlation_matrix = df[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Kahve, Uyku ve Hava Durumu Değişkenleri Arasındaki Korelasyonlar')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/korelasyon_matrisi.png')
    
    # 2. Sıcaklık ve kahve tüketimi
    plt.figure(figsize=(12, 6))
    plt.scatter(df['avg_temp'], df['CupsOfCoffee'], alpha=0.7, s=80, c='teal')
    plt.xlabel('Ortalama Sıcaklık (°C)')
    plt.ylabel('Kahve Fincanı Sayısı')
    plt.title('Sıcaklık ve Kahve Tüketimi İlişkisi')
    plt.grid(True, alpha=0.3)
    
    # Regresyon çizgisi
    import numpy as np
    z = np.polyfit(df['avg_temp'], df['CupsOfCoffee'], 1)
    p = np.poly1d(z)
    plt.plot(df['avg_temp'], p(df['avg_temp']), "r--", linewidth=2)
    
    # Korelasyon değerini ekle
    corr = df['avg_temp'].corr(df['CupsOfCoffee'])
    plt.annotate(f"Korelasyon: {corr:.2f}", xy=(0.05, 0.95), xycoords='axes fraction', 
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/sicaklik_kahve_iliskisi.png')
    
    # 3. Bulut örtüsü ve kahve tüketimi
    plt.figure(figsize=(12, 6))
    plt.scatter(df['cloud_cover'], df['CupsOfCoffee'], alpha=0.7, s=80, c='navy')
    plt.xlabel('Bulut Örtüsü (%)')
    plt.ylabel('Kahve Fincanı Sayısı')
    plt.title('Bulut Örtüsü ve Kahve Tüketimi İlişkisi')
    plt.grid(True, alpha=0.3)
    
    # Regresyon çizgisi
    z = np.polyfit(df['cloud_cover'], df['CupsOfCoffee'], 1)
    p = np.poly1d(z)
    plt.plot(df['cloud_cover'], p(df['cloud_cover']), "r--", linewidth=2)
    
    # Korelasyon değerini ekle
    corr = df['cloud_cover'].corr(df['CupsOfCoffee'])
    plt.annotate(f"Korelasyon: {corr:.2f}", xy=(0.05, 0.95), xycoords='axes fraction', 
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/bulut_kahve_iliskisi.png')
    
    # 4. Yağış ve kahve tüketimi
    plt.figure(figsize=(12, 6))
    plt.scatter(df['precipitation'], df['CupsOfCoffee'], alpha=0.7, s=80, c='darkblue')
    plt.xlabel('Yağış Miktarı (mm)')
    plt.ylabel('Kahve Fincanı Sayısı')
    plt.title('Yağış ve Kahve Tüketimi İlişkisi')
    plt.grid(True, alpha=0.3)
    
    # Regresyon çizgisi (eğer yeterli veri noktası varsa)
    if len(df['precipitation'].unique()) > 3:
        z = np.polyfit(df['precipitation'], df['CupsOfCoffee'], 1)
        p = np.poly1d(z)
        plt.plot(df['precipitation'], p(df['precipitation']), "r--", linewidth=2)
    
    # Korelasyon değerini ekle
    corr = df['precipitation'].corr(df['CupsOfCoffee'])
    plt.annotate(f"Korelasyon: {corr:.2f}", xy=(0.05, 0.95), xycoords='axes fraction', 
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/yagis_kahve_iliskisi.png')
    
    # 5. Zaman içinde kahve ve hava durumu
    plt.figure(figsize=(14, 8))
    
    # İlk grafik: Kahve ve sıcaklık
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(df['Date'], df['CupsOfCoffee'], 'o-', color='brown', label='Kahve (Fincan)')
    ax1.set_ylabel('Kahve Fincanı', color='brown')
    ax1.tick_params(axis='y', labelcolor='brown')
    
    ax2 = ax1.twinx()
    ax2.plot(df['Date'], df['avg_temp'], 's-', color='red', label='Sıcaklık (°C)')
    ax2.set_ylabel('Sıcaklık (°C)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # El ile legend oluştur
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.title('Zaman İçinde Kahve Tüketimi ve Sıcaklık')
    
    # İkinci grafik: Kahve ve bulut örtüsü
    ax3 = plt.subplot(3, 1, 2, sharex=ax1)
    ax3.plot(df['Date'], df['CupsOfCoffee'], 'o-', color='brown', label='Kahve (Fincan)')
    ax3.set_ylabel('Kahve Fincanı', color='brown')
    ax3.tick_params(axis='y', labelcolor='brown')
    
    ax4 = ax3.twinx()
    ax4.plot(df['Date'], df['cloud_cover'], 's-', color='skyblue', label='Bulut Örtüsü (%)')
    ax4.set_ylabel('Bulut Örtüsü (%)', color='skyblue')
    ax4.tick_params(axis='y', labelcolor='skyblue')
    
    # El ile legend oluştur
    lines3, labels3 = ax3.get_legend_handles_labels()
    lines4, labels4 = ax4.get_legend_handles_labels()
    ax3.legend(lines3 + lines4, labels3 + labels4, loc='upper right')
    
    plt.title('Zaman İçinde Kahve Tüketimi ve Bulut Örtüsü')
    
    # Üçüncü grafik: Kahve ve yağış
    ax5 = plt.subplot(3, 1, 3, sharex=ax1)
    ax5.plot(df['Date'], df['CupsOfCoffee'], 'o-', color='brown', label='Kahve (Fincan)')
    ax5.set_ylabel('Kahve Fincanı', color='brown')
    ax5.tick_params(axis='y', labelcolor='brown')
    
    ax6 = ax5.twinx()
    ax6.bar(df['Date'], df['precipitation'], alpha=0.3, color='blue', label='Yağış (mm)')
    ax6.set_ylabel('Yağış (mm)', color='blue')
    ax6.tick_params(axis='y', labelcolor='blue')
    
    # El ile legend oluştur
    lines5, labels5 = ax5.get_legend_handles_labels()
    lines6, labels6 = ax6.get_legend_handles_labels()
    ax5.legend(lines5 + lines6, labels5 + labels6, loc='upper right')
    
    plt.title('Zaman İçinde Kahve Tüketimi ve Yağış')
    plt.xlabel('Tarih')
    
    # Genel ayarlar
    plt.tight_layout()
    plt.savefig(f'{output_dir}/zaman_serisi_analizi.png')
    
    print(f"Görselleştirmeler '{output_dir}' klasörüne kaydedildi.")

def main():
    print("Kahve, Uyku ve Hava Durumu Verileri Birleştirme ve Analiz")
    print("--------------------------------------------------------\n")
    
    # Veri setlerini birleştir
    merged_data = merge_datasets()
    
    # Görselleştirmeler oluştur
    create_visualizations(merged_data)
    
    # Özet istatistikleri göster
    if merged_data is not None and not merged_data.empty:
        print("\nÖzet İstatistikler:")
        
        # Kahve ve hava durumu arasındaki korelasyonları göster
        corr_temp = merged_data['CupsOfCoffee'].corr(merged_data['avg_temp'])
        corr_cloud = merged_data['CupsOfCoffee'].corr(merged_data['cloud_cover'])
        corr_rain = merged_data['CupsOfCoffee'].corr(merged_data['precipitation'])
        
        print(f"Kahve tüketimi - Sıcaklık korelasyonu: {corr_temp:.3f}")
        print(f"Kahve tüketimi - Bulut örtüsü korelasyonu: {corr_cloud:.3f}")
        print(f"Kahve tüketimi - Yağış korelasyonu: {corr_rain:.3f}")
        
        # Sonuçların özeti
        print("\nSonuçların Yorumu:")
        factors = []
        if abs(corr_temp) > 0.3:
            direction = "pozitif" if corr_temp > 0 else "negatif"
            factors.append(f"sıcaklık ({direction} ilişki)")
        if abs(corr_cloud) > 0.3:
            direction = "pozitif" if corr_cloud > 0 else "negatif"
            factors.append(f"bulut örtüsü ({direction} ilişki)")
        if abs(corr_rain) > 0.3:
            direction = "pozitif" if corr_rain > 0 else "negatif"
            factors.append(f"yağış ({direction} ilişki)")
        
        if factors:
            print(f"Kahve tüketimi ile {', '.join(factors)} arasında anlamlı bir ilişki bulunmaktadır.")
        else:
            print("Kahve tüketimi ile hava durumu değişkenleri arasında güçlü bir ilişki bulunamadı.")
        
        print("\nAnaliz tamamlandı.")

if __name__ == "__main__":
    main() 