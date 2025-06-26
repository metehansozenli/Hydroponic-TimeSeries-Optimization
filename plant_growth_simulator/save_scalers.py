import json
import numpy as np
from sklearn.preprocessing import RobustScaler
import pandas as pd

# Bu script, LSTM modelinde kullanılan scalerleri kaydeder
# İnteraktif bitki büyüme simülatörü için gerekli olacak

def save_scalers():
    """
    lstm_plant_growth_model_optimum_env.py dosyasında oluşturulan 
    scaler_X ve scaler_Y nesnelerini bir JSON dosyasına kaydeder.
    """
    # Aynı veri seti ile scalerları yeniden oluştururuz
    DATA_PATH = 'birlesik_veri_pchip_monotonic_interpolated_filled.xlsx'
    df = pd.read_excel(DATA_PATH)
    
    # Sadece çevresel değişkenleri seçelim
    feature_cols_env = ['h2o_temp_C', 'pH', 'EC']
    target_col = 'plant_height_cm'
    
    # RobustScaler'lar oluştur
    scaler_X = RobustScaler()
    scaler_Y = RobustScaler()
    
    # Veriyi hazırlayıp scaler'ları fit edelim
    X_env = df[feature_cols_env].values
    Y = df[target_col].values.reshape(-1, 1)
    
    X_env_scaled = scaler_X.fit_transform(X_env)
    Y_scaled = scaler_Y.fit_transform(Y)
    
    # Scaler parametrelerini JSON formatında kaydet
    scalers_dict = {
        'scaler_X': {
            'center': scaler_X.center_.tolist(),
            'scale': scaler_X.scale_.tolist()
        },
        'scaler_Y': {
            'center': scaler_Y.center_.tolist(),
            'scale': scaler_Y.scale_.tolist()
        }
    }
    
    with open('scalers.json', 'w') as f:
        json.dump(scalers_dict, f)
    
    print("Scalers başarıyla 'scalers.json' dosyasına kaydedildi.")
    
    # İşte scalerleri test edelim
    sample_input = np.array([[22.5, 6.0, 2.5]])  # Örnek çevresel değerler
    sample_scaled = scaler_X.transform(sample_input)
    print(f"Test - Örnek girdi: {sample_input}")
    print(f"Test - Ölçeklenmiş girdi: {sample_scaled}")
    
    # Scaler'ı test etmek için yükleme
    with open('scalers.json', 'r') as f:
        loaded_scalers = json.load(f)
    
    # Yeni bir scaler oluştur ve değerleri yükle
    test_scaler_X = RobustScaler()
    test_scaler_X.center_ = np.array(loaded_scalers['scaler_X']['center'])
    test_scaler_X.scale_ = np.array(loaded_scalers['scaler_X']['scale'])
    
    # Test için aynı veriyi ölçekle
    test_scaled = test_scaler_X.transform(sample_input)
    print(f"Yüklenen scaler ile ölçeklenmiş: {test_scaled}")
    print(f"Doğru çalışıyor mu? {np.allclose(sample_scaled, test_scaled)}")

if __name__ == "__main__":
    save_scalers()
