# Gerçek optimum değerleri al ve doğru model tahminini kullan
import sys
sys.path.append('..')

# Gerçek modeli ve scaler'ları yükle
import json
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import RobustScaler

def load_real_model_and_get_optimums():
    """Ana LSTM dosyasından gerçek optimum değerleri al"""
    try:
        # Model ve scaler'ları yükle
        model = tf.keras.models.load_model('../plant_growth_lstm_model.h5')
        
        with open('../scalers.json', 'r') as f:
            scalers_dict = json.load(f)
        
        scaler_X = RobustScaler()
        scaler_Y = RobustScaler()
        scaler_X.center_ = np.array(scalers_dict['scaler_X']['center'])
        scaler_X.scale_ = np.array(scalers_dict['scaler_X']['scale'])
        scaler_Y.center_ = np.array(scalers_dict['scaler_Y']['center']) 
        scaler_Y.scale_ = np.array(scalers_dict['scaler_Y']['scale'])
        
        return model, scaler_X, scaler_Y, True
    except Exception as e:
        print(f"Model yükleme hatası: {e}")
        return None, None, None, False

def get_real_optimums():
    """Gerçek optimum değerleri ana dosyadan al"""
    try:
        # Ana dosyadan optimum değerleri import et
        from lstm_plant_growth_model_optimum_env import opt_temp, opt_ph, opt_ec, opt_temp_rate, opt_ph_rate, opt_ec_rate
        return {
            'max_height': {'temp': opt_temp, 'ph': opt_ph, 'ec': opt_ec},
            'max_growth_rate': {'temp': opt_temp_rate, 'ph': opt_ph_rate, 'ec': opt_ec_rate}
        }
    except ImportError as e:
        print(f"Optimum değerler import edilemedi: {e}")
        # Varsayılan değerler (modelinizden gözlemlenen)
        return {
            'max_height': {'temp': 23.82, 'ph': 6.35, 'ec': 2.86},
            'max_growth_rate': {'temp': 23.82, 'ph': 6.35, 'ec': 2.86}
        }

if __name__ == "__main__":
    # Test et
    model, scaler_X, scaler_Y, success = load_real_model_and_get_optimums()
    print(f"Model yükleme başarılı: {success}")
    
    optimums = get_real_optimums()
    print(f"Gerçek optimum değerler: {optimums}")
