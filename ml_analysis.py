import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
import joblib

# Stil ayarları
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# Çıktı klasörü oluştur
output_dir = 'ml_analysis_output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"'{output_dir}' klasörü oluşturuldu")

def load_and_enrich_data():
    """Veri setlerini yükle, birleştir ve ML için zenginleştir"""
    print("Veri setleri yükleniyor ve zenginleştiriliyor...")
    
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
    
    # FEATURE ENRICHMENT #
    
    # 1. Uyku saatlerini zenginleştir
    # 1.1 Uyku saatlerini dakikaya çevir
    merged_df['SleepMinutes'] = merged_df['SleepingHours'].apply(lambda x: int(x.split(':')[0])*60 + int(x.split(':')[1]))
    merged_df['SleepHours'] = merged_df['SleepMinutes'] / 60
    
    # 1.2 Uyku kategorileri oluştur (Feature Transformation)
    merged_df['SleepCategory'] = pd.cut(
        merged_df['SleepHours'], 
        bins=[0, 6, 7, 10], 
        labels=['Insufficient', 'Normal', 'Sufficient']
    )
    
    # 1.3 Önceki günün uyku süresini ekle (Feature Enrichment - Time Series)
    merged_df['PreviousDaySleep'] = merged_df['SleepHours'].shift(1)
    
    # 1.4 Uyku trendi (son 3 günün eğilimi) (Feature Enrichment - Time Series)
    merged_df['SleepTrend'] = merged_df['SleepHours'].rolling(window=3).mean().shift(1)
    
    # 2. Zamansal özellikleri zenginleştir
    # 2.1 Haftanın günleri
    merged_df['DayOfWeek'] = merged_df['Date'].dt.day_name()
    merged_df['IsWeekend'] = merged_df['Date'].dt.dayofweek >= 5  # Hafta sonu: True (5=Cumartesi, 6=Pazar)
    
    # 2.2 Hafta numarası 
    merged_df['WeekOfYear'] = merged_df['Date'].dt.isocalendar().week
    
    # 3. Hava durumu özelliklerini zenginleştir
    # 3.1 Sıcaklık kategorileri (Feature Transformation)
    merged_df['TemperatureCategory'] = pd.cut(
        merged_df['avg_temp'], 
        bins=[0, 8, 12, 16, 100], 
        labels=['Cold', 'Mild', 'Warm', 'Hot']
    )
    
    # 3.2 Yağış kategorileri (Feature Transformation)
    merged_df['PrecipitationCategory'] = pd.cut(
        merged_df['precipitation'], 
        bins=[-1, 0.1, 1, 5, 100], 
        labels=['None', 'Light', 'Moderate', 'Heavy']
    )
    
    # 3.3 Bulut örtüsü kategorileri (Feature Transformation & Cleaning)
    # Not: Veri setinde negatif değerler var, bunlar hata olmalı
    merged_df['cloud_cover'] = merged_df['cloud_cover'].apply(lambda x: max(0, min(x, 100)))  # 0-100 arasına sınırla
    merged_df['CloudCategory'] = pd.cut(
        merged_df['cloud_cover'], 
        bins=[0, 25, 50, 75, 100], 
        labels=['Clear', 'Partly Cloudy', 'Mostly Cloudy', 'Overcast']
    )
    
    # 3.4 Hava durumu skoru (Feature Engineering - Composite Feature)
    # Sıcaklık + nem + bulut örtüsü birleşimi (kahve içme olasılığına etki eden faktör)
    # Sıcaklık: düşük=yüksek skor, Bulut örtüsü: yüksek=yüksek skor
    # Düşük temp, yüksek bulut örtüsü = yüksek kahve isteği
    merged_df['WeatherScore'] = (
        (20 - merged_df['avg_temp']) * 0.5 +  # Düşük sıcaklık → yüksek puan 
        merged_df['cloud_cover'] * 0.3 +       # Yüksek bulut → yüksek puan
        merged_df['precipitation'] * 2         # Yağış → yüksek puan
    )
    
    # 4. Akademik stres skorunu zenginleştir (Feature Engineering)
    # 4.1 Son 3 günde yaklaşan sınav/ödev sayısı
    merged_df['UpcomingAcademicEvents'] = merged_df['DoYouHaveAnySubmissionOrExam'].rolling(window=3).sum().shift(-2)
    merged_df['UpcomingAcademicEvents'] = merged_df['UpcomingAcademicEvents'].fillna(0)
    
    # 4.2 Yakın geçmişteki sınav/ödev sayısı (son 2 gün)
    merged_df['RecentAcademicEvents'] = merged_df['DoYouHaveAnySubmissionOrExam'].rolling(window=2).sum().shift(1)
    merged_df['RecentAcademicEvents'] = merged_df['RecentAcademicEvents'].fillna(0)
    
    # 5. Kahve tüketim geçmişini zenginleştir (Feature Enrichment - Time Series)
    # 5.1 Önceki gün kahve tüketimi
    merged_df['PreviousDayCoffee'] = merged_df['CupsOfCoffee'].shift(1)
    
    # 5.2 Son 3 günlük kahve tüketim ortalaması
    merged_df['CoffeeTrend'] = merged_df['CupsOfCoffee'].rolling(window=3).mean().shift(1)
    
    # 6. Bileşik stres skoru (Feature Engineering - Composite Feature)
    merged_df['StressScore'] = (
        (7 - merged_df['SleepHours']) * 3 +                # Az uyku → yüksek stres
        merged_df['UpcomingAcademicEvents'] * 2 +          # Yaklaşan akademik etkinlikler → yüksek stres
        merged_df['RecentAcademicEvents'] * 1 +            # Geçmiş akademik etkinlikler → orta stres
        (merged_df['WeatherScore'] / 10)                   # Kötü hava → biraz stres
    )
    
    # NaN değerlerini doldur
    merged_df = merged_df.fillna({
        'PreviousDaySleep': merged_df['SleepHours'].mean(),
        'SleepTrend': merged_df['SleepHours'].mean(),
        'PreviousDayCoffee': merged_df['CupsOfCoffee'].mean(),
        'CoffeeTrend': merged_df['CupsOfCoffee'].mean(),
        'UpcomingAcademicEvents': 0,
        'RecentAcademicEvents': 0
    })
    
    # Zenginleştirilmiş veri setini kaydet
    enriched_file = f'{output_dir}/enriched_coffee_data.csv'
    merged_df.to_csv(enriched_file, index=False)
    print(f"Zenginleştirilmiş veri seti {enriched_file} dosyasına kaydedildi.")
    
    # Eklenen/dönüştürülen özelliklerin özetini göster
    print("\nZenginleştirilmiş Özellikler:")
    new_features = [
        'SleepHours', 'SleepCategory', 'PreviousDaySleep', 'SleepTrend',
        'DayOfWeek', 'IsWeekend', 'WeekOfYear',
        'TemperatureCategory', 'PrecipitationCategory', 'CloudCategory', 'WeatherScore',
        'UpcomingAcademicEvents', 'RecentAcademicEvents',
        'PreviousDayCoffee', 'CoffeeTrend', 'StressScore'
    ]
    
    for feature in new_features:
        if feature in merged_df.columns:
            # Check if the feature is categorical
            if merged_df[feature].dtype.name == 'category':
                print(f"- {feature}: {merged_df[feature].cat.categories.tolist()}")
            # Check if the feature is boolean
            elif merged_df[feature].dtype == bool:
                print(f"- {feature}: Boolean feature")
            # Check if the feature is string type
            elif merged_df[feature].dtype == object:
                unique_values = merged_df[feature].unique()
                if len(unique_values) <= 10:  # If there are few unique values, show them
                    print(f"- {feature}: {list(unique_values)}")
                else:
                    print(f"- {feature}: Text feature with {len(unique_values)} unique values")
            # For numeric features, show min, max, mean
            else:
                print(f"- {feature}: min={merged_df[feature].min():.2f}, max={merged_df[feature].max():.2f}, mean={merged_df[feature].mean():.2f}")
    
    return merged_df

def visualize_enriched_features(df):
    """Zenginleştirilmiş özellikleri görselleştir"""
    print("\nZenginleştirilmiş özelliklerin görselleştirmeleri oluşturuluyor...")
    
    # 1. Stres skoru ve kahve tüketimi ilişkisi
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='StressScore', y='CupsOfCoffee', hue='IsWeekend', 
                    size='CupsOfCoffee', sizes=(50, 200), 
                    palette='viridis', data=df)
    
    # Regresyon çizgisi
    sns.regplot(x='StressScore', y='CupsOfCoffee', data=df, scatter=False, 
                line_kws={"color":"red", "alpha":0.7, "lw":2, "ls":"--"})
    
    plt.xlabel('Stres Skoru')
    plt.ylabel('Kahve Fincanı Sayısı')
    plt.title('Stres Skoru ve Kahve Tüketimi İlişkisi')
    
    # Korelasyon değerini ekle
    corr = df['StressScore'].corr(df['CupsOfCoffee'])
    plt.annotate(f"Korelasyon: {corr:.2f}", xy=(0.05, 0.95), xycoords='axes fraction', 
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.grid(True, alpha=0.3)
    plt.legend(title='Hafta Sonu')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/stres_kahve_iliskisi.png')
    
    # 2. Kahve tüketiminin tüm kategorik değişkenlerle ilişkisi
    categorical_features = ['SleepCategory', 'TemperatureCategory', 'PrecipitationCategory', 
                           'CloudCategory', 'DayOfWeek']
    
    fig, axes = plt.subplots(len(categorical_features), 1, figsize=(12, 4*len(categorical_features)))
    
    for i, feature in enumerate(categorical_features):
        if feature in df.columns:
            sns.boxplot(x=feature, y='CupsOfCoffee', data=df, ax=axes[i])
            axes[i].set_title(f'{feature} ve Kahve Tüketimi İlişkisi')
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel('Kahve Fincanı Sayısı')
            axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/kategorik_degiskenler_kahve_iliskisi.png')
    
    # 3. Özellik korelasyon matrisi
    numeric_cols = [
        'CupsOfCoffee', 'SleepHours', 'avg_temp', 'precipitation', 
        'cloud_cover', 'PreviousDaySleep', 'SleepTrend', 'WeatherScore',
        'UpcomingAcademicEvents', 'RecentAcademicEvents', 'PreviousDayCoffee',
        'CoffeeTrend', 'StressScore'
    ]
    
    numeric_cols = [col for col in numeric_cols if col in df.columns]
    
    plt.figure(figsize=(14, 10))
    correlation_matrix = df[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Zenginleştirilmiş Özelliklerin Korelasyon Matrisi')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/zenginlestirilmis_korelasyon_matrisi.png')
    
def prepare_data_for_ml(df):
    """Veriyi ML için hazırla: X ve y ayrımı, train-test split"""
    print("\nVeri ML için hazırlanıyor...")
    
    # Hedef değişken
    y = df['CupsOfCoffee']
    
    # Kategorik ve sayısal özellikler
    categorical_features = ['SleepCategory', 'DayOfWeek', 'IsWeekend', 
                            'TemperatureCategory', 'PrecipitationCategory', 'CloudCategory']
    
    numerical_features = ['SleepHours', 'PreviousDaySleep', 'SleepTrend', 
                           'avg_temp', 'precipitation', 'cloud_cover', 'WeatherScore',
                           'UpcomingAcademicEvents', 'RecentAcademicEvents',
                           'PreviousDayCoffee', 'CoffeeTrend', 'StressScore']
    
    # Mevcut sütunları kontrol et
    categorical_features = [f for f in categorical_features if f in df.columns]
    numerical_features = [f for f in numerical_features if f in df.columns]
    
    # Özellikler
    X = df[categorical_features + numerical_features]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    
    print(f"Eğitim seti boyutu: {X_train.shape}")
    print(f"Test seti boyutu: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, categorical_features, numerical_features

def build_and_evaluate_models(X_train, X_test, y_train, y_test, categorical_features, numerical_features):
    """ML modellerini oluştur, eğit ve değerlendir"""
    print("\nML modelleri oluşturuluyor ve değerlendiriliyor...")
    
    # Preprocessing için veri dönüşüm pipeline'ları oluştur
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Kategorik ve sayısal özellikler için birleşik preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    # Model tanımları
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.1),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "Support Vector Regression": SVR(kernel='rbf')
    }
    
    # Her model için pipeline oluştur, eğit ve değerlendir
    results = {}
    feature_importances = {}
    
    for name, model in models.items():
        print(f"\nModel: {name} eğitiliyor ve değerlendiriliyor...")
        
        # Pipeline oluştur
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        # Model eğitimi
        pipeline.fit(X_train, y_train)
        
        # Tahminler
        y_pred = pipeline.predict(X_test)
        
        # Performans metrikleri
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Sonuçları kaydet
        results[name] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
        
        print(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
        
        # Özellik önemleri (destekleyen modeller için)
        if hasattr(model, 'feature_importances_'):
            # Preprocessor'dan sonra özellik isimleri
            preprocessor.fit(X_train)
            
            # Kategorik değişkenler için one-hot-encoding sonrası isimler
            cat_features = []
            if categorical_features:
                cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
                cat_features = cat_encoder.get_feature_names_out(categorical_features)
            
            # Tüm özellik isimleri
            feature_names = np.concatenate([numerical_features, cat_features])
            
            # Özellik önemleri
            importances = model.feature_importances_
            
            # En önemli 10 özelliği depola
            indices = np.argsort(importances)[::-1][:min(10, len(feature_names))]
            top_features = {feature_names[i]: importances[i] for i in indices}
            feature_importances[name] = top_features
    
    # En iyi modeli seç
    best_model = max(results, key=lambda k: results[k]['R2'])
    print(f"\nEn iyi model: {best_model} (R2: {results[best_model]['R2']:.4f})")
    
    # Sonuçları görselleştir
    # 1. Model performans karşılaştırması
    plt.figure(figsize=(12, 6))
    model_names = list(results.keys())
    r2_scores = [results[model]['R2'] for model in model_names]
    rmse_scores = [results[model]['RMSE'] for model in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    fig, ax1 = plt.subplots(figsize=(14, 7))
    bar1 = ax1.bar(x - width/2, r2_scores, width, label='R² Skoru', color='steelblue')
    ax1.set_ylabel('R² Skoru')
    ax1.set_ylim(0, 1)
    
    ax2 = ax1.twinx()
    bar2 = ax2.bar(x + width/2, rmse_scores, width, label='RMSE', color='coral')
    ax2.set_ylabel('RMSE')
    
    ax1.set_xlabel('Model')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    plt.title('ML Modelleri Performans Karşılaştırması')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/model_performans_karsilastirmasi.png')
    
    # 2. En iyi modelin özellik önemleri (varsa)
    if best_model in feature_importances:
        plt.figure(figsize=(12, 6))
        features = list(feature_importances[best_model].keys())
        importances = list(feature_importances[best_model].values())
        
        sns.barplot(x=importances, y=features)
        plt.title(f'{best_model} - Özellik Önemleri')
        plt.xlabel('Önem')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/en_iyi_model_ozellik_onemleri.png')
    
    # En iyi modeli kaydet
    best_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', models[best_model])
    ])
    best_pipeline.fit(X_train, y_train)
    
    joblib.dump(best_pipeline, f'{output_dir}/best_coffee_prediction_model.pkl')
    print(f"En iyi model '{output_dir}/best_coffee_prediction_model.pkl' dosyasına kaydedildi.")
    
    return results, best_model

def feature_selection_analysis(X_train, y_train, categorical_features, numerical_features):
    """Özellik seçimi analizi - RFE kullanarak"""
    print("\nÖzellik seçimi analizi yapılıyor...")
    
    # Preprocessing pipeline
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    # Veriyi dönüştür
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    
    # Özellik isimleri
    cat_features = []
    if categorical_features:
        cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_features = cat_encoder.get_feature_names_out(categorical_features)
    
    feature_names = np.concatenate([numerical_features, cat_features])
    
    # RFE için model
    estimator = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Farklı özellik sayıları için RFE
    n_features_to_select_options = [3, 5, 7, 10, 15]
    results = {}
    
    for n in n_features_to_select_options:
        if n <= X_train_preprocessed.shape[1]:
            # RFE uygula
            selector = RFE(estimator, n_features_to_select=n, step=1)
            selector = selector.fit(X_train_preprocessed, y_train)
            
            # Seçilen özellikler
            selected_features_mask = selector.support_
            selected_features = feature_names[selected_features_mask]
            
            results[n] = {
                'selected_features': selected_features,
                'ranking': selector.ranking_
            }
    
    # Sonuçları yazdır
    print("\nÖzellik Seçimi Sonuçları:")
    for n, result in results.items():
        print(f"\nEn önemli {n} özellik:")
        for feature in result['selected_features']:
            print(f"- {feature}")
    
    # Sonuçları görselleştir
    if results:
        # Tüm özelliklerin önem sıralaması
        plt.figure(figsize=(12, 8))
        
        # RFE son durumdan sıralama
        last_n = max(n_features_to_select_options)
        if last_n in results:
            rankings = results[last_n]['ranking']
            
            # Özellikleri ve sıralamalarını bir dictionary'e koy
            feature_rankings = {str(feature): rank for feature, rank in zip(feature_names, rankings)}
            
            # Sıralamaya göre sırala (düşük değerler daha önemli)
            sorted_features = sorted(feature_rankings.items(), key=lambda x: x[1])
            
            features = [feature for feature, _ in sorted_features]
            ranks = [rank for _, rank in sorted_features]
            
            plt.figure(figsize=(10, 12))
            sns.barplot(x=ranks, y=features)
            plt.title('Özellik Önem Sıralaması (RFE)')
            plt.xlabel('Sıralama (düşük=önemli)')
            plt.ylabel('Özellik')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/ozellik_siralama_rfe.png')
    
    return results

def main():
    """Ana fonksiyon: Tüm analizleri çalıştırır"""
    print("Kahve Tüketimi Makine Öğrenmesi Analizi")
    print("========================================\n")
    
    # Verileri yükle ve zenginleştir
    df = load_and_enrich_data()
    
    if df is None or df.empty:
        print("Hata: Veri yüklenemedi veya zenginleştirilemedi.")
        return
    
    # Zenginleştirilmiş özellikleri görselleştir
    visualize_enriched_features(df)
    
    # Veriyi ML için hazırla
    X_train, X_test, y_train, y_test, categorical_features, numerical_features = prepare_data_for_ml(df)
    
    # Özellik seçimi analizi
    feature_selection_results = feature_selection_analysis(X_train, y_train, categorical_features, numerical_features)
    
    # ML modellerini oluştur ve değerlendir
    model_results, best_model = build_and_evaluate_models(
        X_train, X_test, y_train, y_test, 
        categorical_features, numerical_features
    )
    
    print(f"\nML analizi tamamlandı. Tüm görseller ve modeller '{output_dir}' klasörüne kaydedildi.")

if __name__ == "__main__":
    main() 