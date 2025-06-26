import numpy as np
import pandas as pd
import json
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from io import BytesIO
import base64
import os

class GrowthPredictor:    
    def __init__(self, model_path='../plant_growth_lstm_model.h5', 
                 scaler_path='../scalers.json', 
                 optimum_path='../optimum_values.json'):
        """Initialize the growth predictor with the trained LSTM model and scalers."""
        
        print("Gerçek LSTM modelini yükleniyor...")
        
        # Model yükleme
        try:
            self.model = load_model(model_path, compile=False)
            # Modeli yeniden compile et
            self.model.compile(optimizer='adam', loss='mse')
            print("LSTM model başarıyla yüklendi!")
            self.use_real_model = True
        except Exception as e:
            print(f"LSTM model yükleme hatası: {e}")
            raise Exception("LSTM model yüklenemedi! Lütfen önce modeli eğitin.")
        
        # Scaler'ları yükle
        try:
            with open(scaler_path, 'r') as f:
                scalers_dict = json.load(f)
                
            self.scaler_X = RobustScaler()
            self.scaler_Y = RobustScaler()
            
            self.scaler_X.center_ = np.array(scalers_dict['scaler_X']['center'])
            self.scaler_X.scale_ = np.array(scalers_dict['scaler_X']['scale'])
            self.scaler_Y.center_ = np.array(scalers_dict['scaler_Y']['center'])
            self.scaler_Y.scale_ = np.array(scalers_dict['scaler_Y']['scale'])
            
            print("Scaler'lar başarıyla yüklendi!")
        except Exception as e:
            print(f"Scaler yükleme hatası: {e}")
            # Varsayılan değerler
            self.scaler_X = RobustScaler()
            self.scaler_Y = RobustScaler()
            self.scaler_X.center_ = np.array([22.5, 6.3, 2.5])
            self.scaler_X.scale_ = np.array([2.0, 0.5, 0.5])
            self.scaler_Y.center_ = np.array([30.0])
            self.scaler_Y.scale_ = np.array([10.0])
        
        # Optimum değerleri yükle
        try:
            with open(optimum_path, 'r') as f:
                optimum_dict = json.load(f)
            
            self.optimal_temp = optimum_dict['optimal_temp']
            self.optimal_ph = optimum_dict['optimal_ph']
            self.optimal_ec = optimum_dict['optimal_ec']
            self.optimal_growth = optimum_dict['optimal_growth']
            self.optimal_temp_rate = optimum_dict.get('optimal_temp_rate', optimum_dict['optimal_temp'])
            self.optimal_ph_rate = optimum_dict.get('optimal_ph_rate', optimum_dict['optimal_ph'])
            self.optimal_ec_rate = optimum_dict.get('optimal_ec_rate', optimum_dict['optimal_ec'])
            self.optimal_growth_rate = optimum_dict.get('optimal_growth_rate', 1.0)
            self.bounds = optimum_dict['bounds']
            self.time_steps = optimum_dict['time_steps']
            self.feature_cols_env = optimum_dict['feature_cols_env']
            
            print("Optimum değerler başarıyla yüklendi!")
            print(f"Optimum sıcaklık: {self.optimal_temp:.2f}°C")
            print(f"Optimum pH: {self.optimal_ph:.2f}")
            print(f"Optimum EC: {self.optimal_ec:.2f}")
            
        except Exception as e:
            print(f"Optimum değerler yükleme hatası: {e}")
            # Varsayılan değerler
            self.optimal_temp = 23.82
            self.optimal_ph = 6.35
            self.optimal_ec = 2.86
            self.optimal_growth = 50.0
            self.optimal_temp_rate = 23.82
            self.optimal_ph_rate = 6.35
            self.optimal_ec_rate = 2.86
            self.optimal_growth_rate = 1.0
            self.bounds = {
                'h2o_temp_C': (19, 27),
                'pH': (5.4, 7.4),
                'EC': (1.6, 3.8)
            }
            self.time_steps = 4
            self.feature_cols_env = ['h2o_temp_C', 'pH', 'EC']
    
    def predict_growth(self, h2o_temp, ph, ec, days=30):
        """
        Predict plant growth over specified number of days with given parameters.
        LSTM modeli zaman serisi formatında çevresel verileri kullanarak bitki boyunu tahmin eder.
        
        Args:
            h2o_temp: Water temperature in Celsius
            ph: pH value
            ec: EC value in mS/cm
            days: Number of days to simulate growth
            
        Returns:
            Dictionary with predicted heights and growth rate assessment
        """
        heights = []
        
        # Başlangıç sequence'i oluştur (time_steps uzunluğunda)
        # İlk birkaç gün için aynı çevresel koşulları kullan
        env_sequence = []
        for i in range(self.time_steps):
            # Hafif varyasyonlar ekle
            temp_var = np.random.normal(0, 0.1)
            ph_var = np.random.normal(0, 0.02)
            ec_var = np.random.normal(0, 0.05)
            
            env_sequence.append([
                h2o_temp + temp_var,
                ph + ph_var,
                ec + ec_var
            ])
        
        # İlk tahmin
        initial_height = self._predict_with_lstm_sequence(np.array(env_sequence))
        heights.append(float(initial_height))
        
        # Günlük tahminler
        for day in range(days):
            # Yeni günün çevresel koşulları (küçük varyasyonlar ile)
            temp_var = np.random.normal(0, 0.3)
            ph_var = np.random.normal(0, 0.05)
            ec_var = np.random.normal(0, 0.08)
            
            new_env = [
                np.clip(h2o_temp + temp_var, self.bounds['h2o_temp_C'][0], self.bounds['h2o_temp_C'][1]),
                np.clip(ph + ph_var, self.bounds['pH'][0], self.bounds['pH'][1]),
                np.clip(ec + ec_var, self.bounds['EC'][0], self.bounds['EC'][1])
            ]
            
            # Sequence'i güncelle (sliding window)
            env_sequence = env_sequence[1:] + [new_env]
            
            # LSTM ile tahmin
            predicted_height = self._predict_with_lstm_sequence(np.array(env_sequence))
            
            # Gerçekçi büyüme kontrolü
            if predicted_height > heights[-1]:
                # Büyüme var
                growth = predicted_height - heights[-1]
                # Maksimum günlük büyüme sınırı
                max_daily_growth = 1.5
                actual_growth = min(growth, max_daily_growth)
                new_height = heights[-1] + actual_growth
            else:
                # Model küçüleceğini söylüyorsa, çok küçük bir büyüme ekle
                new_height = heights[-1] + np.random.uniform(0.05, 0.15)
            
            heights.append(float(new_height))
        
        # Parameter quality score
        temp_score = 1 - min(abs(h2o_temp - self.optimal_temp) / 
                            (self.bounds['h2o_temp_C'][1] - self.bounds['h2o_temp_C'][0]), 1)
        ph_score = 1 - min(abs(ph - self.optimal_ph) / 
                          (self.bounds['pH'][1] - self.bounds['pH'][0]), 1)
        ec_score = 1 - min(abs(ec - self.optimal_ec) / 
                          (self.bounds['EC'][1] - self.bounds['EC'][0]), 1)
        overall_score = (temp_score + ph_score + ec_score) / 3
        
        # Create assessment based on the score
        if overall_score >= 0.9:
            assessment = "Mükemmel! Bu koşullar bitkinin optimum büyümesi için ideal."
        elif overall_score >= 0.75:
            assessment = "Çok İyi: Bu koşullar bitkinin sağlıklı büyümesini destekliyor."
        elif overall_score >= 0.6:
            assessment = "İyi: Bu koşullar bitki büyümesi için yeterli."
        elif overall_score >= 0.4:
            assessment = "Orta: Bu koşullar altında büyüme yavaşlayabilir."
        else:
            assessment = "Zayıf: Bu koşullar optimum büyümeyi desteklemiyor."
            
        final_growth_rate = heights[-1] - heights[-2] if len(heights) > 1 else 0
        
        return {
            'heights': [float(h) for h in heights],
            'days': list(range(days + 1)),
            'assessment': assessment,
            'comparison': {
                'temperature': {
                    'value': float(h2o_temp),
                    'optimal': float(self.optimal_temp),
                    'score': float(temp_score * 100)
                },
                'ph': {
                    'value': float(ph),
                    'optimal': float(self.optimal_ph),
                    'score': float(ph_score * 100)
                },
                'ec': {
                    'value': float(ec),
                    'optimal': float(self.optimal_ec),
                    'score': float(ec_score * 100)
                },
                'overall_score': float(overall_score * 100)
            },
            'growth_rate': float(final_growth_rate)
        }
    
    def _predict_with_lstm_sequence(self, env_sequence):
        """LSTM modeli ile sequence tabanlı tahmin."""
        try:
            # Çevresel değişkenleri ölçekle
            env_seq_scaled = self.scaler_X.transform(env_sequence)
            model_input = env_seq_scaled.reshape(1, self.time_steps, len(self.feature_cols_env))
            
            # Model ile tahmin
            pred_scaled = self.model.predict(model_input, verbose=0)
            pred_height = self.scaler_Y.inverse_transform(pred_scaled)[0][0]
            
            # Negatif değerleri önle
            return float(max(pred_height, 5.0))
            
        except Exception as e:
            print(f"LSTM tahmin hatası: {e}")
            return 15.0  # Fallback değer
        
    def generate_growth_chart(self, growth_data):
        """Generate a base64 encoded PNG image of the plant growth chart."""
        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        # Plot the growth curve
        ax.plot(growth_data['days'], growth_data['heights'], 'g-', linewidth=2, label='Seçilen Koşullar')
        ax.set_xlabel('Gün')
        ax.set_ylabel('Bitki Boyu (cm)')
        ax.set_title('Tahmin Edilen Bitki Büyüme Eğrisi')
        ax.grid(True, alpha=0.3)
        
        # Add comparison line for optimal growth
        try:
            optimal_heights = self.predict_growth(
                self.optimal_temp, self.optimal_ph, self.optimal_ec, len(growth_data['days']) - 1
            )['heights']
            ax.plot(growth_data['days'], optimal_heights, 'r--', linewidth=1.5, alpha=0.7, 
                    label='Optimal Koşullarda Büyüme')
        except:
            # Fallback için basit optimal büyüme çizgisi
            optimal_heights = [10 + i * 1.2 for i in range(len(growth_data['days']))]
            ax.plot(growth_data['days'], optimal_heights, 'r--', linewidth=1.5, alpha=0.7, 
                    label='Optimal Koşullarda Büyüme')
        
        ax.legend()
        fig.tight_layout()
        
        # Convert to base64 string
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        
        return img_str
        
    def evaluate_conditions(self, h2o_temp, ph, ec):
        """Evaluate how good the selected conditions are compared to optimal."""
        # Calculate percentage of optimality for each parameter
        temp_range = self.bounds['h2o_temp_C'][1] - self.bounds['h2o_temp_C'][0]
        ph_range = self.bounds['pH'][1] - self.bounds['pH'][0]
        ec_range = self.bounds['EC'][1] - self.bounds['EC'][0]
        
        temp_score = 1 - min(abs(h2o_temp - self.optimal_temp) / temp_range, 1)
        ph_score = 1 - min(abs(ph - self.optimal_ph) / ph_range, 1)
        ec_score = 1 - min(abs(ec - self.optimal_ec) / ec_range, 1)
        
        # Generate advice for each parameter
        temp_advice = "İdeal" if temp_score > 0.9 else "Yüksek" if h2o_temp > self.optimal_temp else "Düşük"
        ph_advice = "İdeal" if ph_score > 0.9 else "Yüksek" if ph > self.optimal_ph else "Düşük"
        ec_advice = "İdeal" if ec_score > 0.9 else "Yüksek" if ec > self.optimal_ec else "Düşük"
        
        return {
            'temperature': {'score': float(temp_score * 100), 'advice': temp_advice},
            'ph': {'score': float(ph_score * 100), 'advice': ph_advice},
            'ec': {'score': float(ec_score * 100), 'advice': ec_advice},
            'overall': float((temp_score + ph_score + ec_score) * 100 / 3)
        }
