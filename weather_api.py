import requests
import pandas as pd
import time
from datetime import datetime, timedelta
import json
import os

def get_weather_data(start_date, end_date, location="Istanbul", api_key=None):
    """
    İstanbul için belirtilen tarihler arasındaki hava durumu verilerini çeker.
    
    Not: OpenWeatherMap API'si normalde maksimum 7 günlük tahmin verir, bu yüzden
    bu script kavramsal olarak doğrudur, ancak gerçek uygulamada, uzun vadeli
    tahminler için farklı bir yaklaşım gerekebilir.
    
    Args:
        start_date (str): Başlangıç tarihi (YYYY-MM-DD formatında)
        end_date (str): Bitiş tarihi (YYYY-MM-DD formatında)
        location (str): Şehir adı (varsayılan: Istanbul)
        api_key (str): OpenWeatherMap API anahtarı
    
    Returns:
        DataFrame: Hava durumu verilerini içeren DataFrame
    """
    # API anahtarının kontrolü
    if api_key is None:
        api_key = input("OpenWeatherMap API anahtarınızı giriniz: ")
    
    # Istanbul koordinatları
    lat = 41.0082
    lon = 28.9784
    
    # Başlangıç ve bitiş tarihlerini datetime nesnelerine dönüştür
    start = datetime.strptime(start_date, "%d.%m.%Y")
    end = datetime.strptime(end_date, "%d.%m.%Y")
    
    # İki tarih arasındaki tüm günleri içeren liste oluştur
    date_range = []
    current_date = start
    while current_date <= end:
        date_range.append(current_date.strftime("%Y-%m-%d"))
        current_date += timedelta(days=1)
    
    # Veri saklama listesi
    weather_data = []
    
    # Örnek veriler ve simülasyon seçeneği
    use_simulation = input("OpenWeatherMap API yerine simülasyon verisi kullanmak ister misiniz? (e/h): ").lower() == 'e'
    
    if use_simulation:
        print(f"Simülasyon verisi oluşturuluyor ({start_date} - {end_date})...")
        import numpy as np
        
        # Hava durumu için gerçekçi değerler
        # İstanbul Mart-Nisan sıcaklıkları yaklaşık 8-18°C arasında olur
        temp_min = 8
        temp_max = 18
        
        for date in date_range:
            # Gerçekçi hava durumu simülasyonu oluştur
            day_of_year = datetime.strptime(date, "%Y-%m-%d").timetuple().tm_yday
            
            # Sıcaklık: Mart başından Nisan sonuna doğru artan bir trend
            progress = (day_of_year - 70) / 90  # 70: Mart başı (yaklaşık), 90: Mart-Nisan (gün sayısı)
            base_temp = temp_min + progress * (temp_max - temp_min)
            random_factor = np.random.normal(0, 2)  # Günlük dalgalanmalar
            temp = round(base_temp + random_factor, 1)
            
            # Yağış: Mart-Nisan döneminde İstanbul'da ortalama 10-15 yağışlı gün olur
            # Yaklaşık %30 olasılıkla yağış
            precipitation = round(np.random.exponential(2) if np.random.random() < 0.3 else 0, 1)
            
            # Bulut örtüsü: Yağış varsa daha yüksek bulut örtüsü
            cloud_cover = min(100, int(np.random.normal(50, 30) + precipitation * 5))
            
            # Veriyi sakla
            formatted_date = datetime.strptime(date, "%Y-%m-%d").strftime("%d.%m.%Y")
            weather_data.append({
                'date': formatted_date,
                'avg_temp': temp,
                'precipitation': precipitation,
                'cloud_cover': cloud_cover
            })
    else:
        # Gerçek API çağrısı için kod
        print(f"OpenWeatherMap API'sinden veri çekiliyor ({start_date} - {end_date})...")
        
        # OpenWeatherMap API'si genellikle 7 günden fazla tahmin vermez
        # Bu yüzden, API'ye günlük çağrılar yapmak en iyisidir
        base_url = "https://api.openweathermap.org/data/2.5/onecall"
        
        for date in date_range:
            params = {
                'lat': lat,
                'lon': lon,
                'exclude': 'current,minutely,hourly,alerts',
                'appid': api_key,
                'units': 'metric',
                'dt': int(datetime.strptime(date, "%Y-%m-%d").timestamp())  # Unix timestamp
            }
            
            try:
                response = requests.get(base_url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    
                    # Daily forecast verisini çek
                    for day in data.get('daily', []):
                        day_date = datetime.fromtimestamp(day['dt']).strftime("%d.%m.%Y")
                        
                        if day_date in [datetime.strptime(d, "%Y-%m-%d").strftime("%d.%m.%Y") for d in date_range]:
                            weather_data.append({
                                'date': day_date,
                                'avg_temp': day['temp']['day'],
                                'precipitation': day.get('rain', 0) if 'rain' in day else 0,
                                'cloud_cover': day['clouds']
                            })
                else:
                    print(f"API Hatası: {response.status_code} - {response.text}")
                
                # API limitlerini aşmamak için bekle
                time.sleep(1)
                
            except Exception as e:
                print(f"Hata: {e}")
    
    # DataFrame oluştur
    weather_df = pd.DataFrame(weather_data)
    
    # CSV'ye kaydet
    output_file = 'raw_datas/weather_data_istanbul.csv'
    weather_df.to_csv(output_file, index=False)
    print(f"Hava durumu verileri {output_file} dosyasına kaydedildi.")
    
    # JSON formatına da kaydet
    with open('raw_datas/weather_data_istanbul.json', 'w') as f:
        json.dump(weather_data, f, indent=4)
    
    return weather_df

if __name__ == "__main__":
    print("İstanbul için Hava Durumu Veri Çekme Aracı")
    print("------------------------------------------\n")
    
    start_date = "11.03.2025"
    end_date = "23.04.2025"
    
    print(f"Tarih Aralığı: {start_date} - {end_date}")
    print("Konum: İstanbul, Türkiye\n")
    
    # API anahtarı çevre değişkeninden al veya kullanıcıdan iste
    api_key = os.environ.get('OPENWEATHERMAP_API_KEY')
    
    # Hava durumu verisini çek
    df = get_weather_data(start_date, end_date, api_key=api_key)
    
    # Özet istatistikler
    print("\nÖzet İstatistikler:")
    print(f"Toplam gün sayısı: {len(df)}")
    print(f"Ortalama sıcaklık: {df['avg_temp'].mean():.1f}°C")
    print(f"Ortalama bulut örtüsü: {df['cloud_cover'].mean():.1f}%")
    print(f"Toplam yağış: {df['precipitation'].sum():.1f} mm")
    
    print("\nVeriler başarıyla kaydedildi. Kahve tüketimi analizinizde kullanabilirsiniz.") 