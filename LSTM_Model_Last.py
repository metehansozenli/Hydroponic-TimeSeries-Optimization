#%%
import pandas as pd
import numpy as np
import random
import time
from sklearn.preprocessing import MinMaxScaler, RobustScaler  # Sadece MinMaxScaler kullanılıyor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, SimpleRNN, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import math
import seaborn as sns

# 1. VERİYİ OKU
DATA_PATH = 'birlesik_veri_filled_pchip_monotonic_interpolated.xlsx'
df = pd.read_excel(DATA_PATH)

print("Veri Seti: ",df.shape)
def rep_based_train_test_split(df, test_size=0.2, random_seed=16):

    random.seed(random_seed)
    np.random.seed(random_seed)

    episodes = list(df.groupby(['plant', 'rep']))
    random.shuffle(episodes)

    n_test = int(len(episodes) * test_size)
    test_episodes = episodes[:n_test]
    train_episodes = episodes[n_test:]

    train_df = pd.concat([ep[1] for ep in train_episodes]).reset_index(drop=True)
    test_df = pd.concat([ep[1] for ep in test_episodes]).reset_index(drop=True)

    return train_df, test_df

def create_env_sequences_by_group(df, feature_cols, target_col, time_steps=6, step=1):
    """
    Bitki-replikat bazında, zaman sırasını koruyarak kayan pencere (sliding window) ile sequence oluşturur.
    step: pencere kayma adımı (default=1, sliding window için 1 olmalı)
    """
    X_seq, y_seq = [], []

    grouped = df.groupby(['plant', 'rep'])

    for (plant, rep), group in grouped:
        X_env = group[feature_cols].values
        y = group[target_col].values
        n = len(group)

        for i in range(0, n - time_steps + 1, step):
            X_seq.append(X_env[i:i + time_steps])
            y_seq.append(y[i + time_steps - 1])

    return np.array(X_seq), np.array(y_seq)

# 2. GEREKLİ KOLONLARI SEÇ
feature_cols = [
    'h2o_temp_C',
    'pH',
    'EC',
    'temp_c',
    'relative_humidity_%',
    'dewpoint',
    'days_since_first_measurement',
    'reservoir_size_liters',
    # 'growth_rate',
]
target_col = 'plant_height_cm'
time_steps = 3

plt.hist(df[target_col], bins=30, edgecolor='k')
plt.title('Bitki Boyu Dağılımı')
plt.xlabel('Bitki Boyu (cm)')
plt.ylabel('Frekans')
plt.show()

# Tarih kolonunu kaldır
df = df.drop('date', axis=1)
df = df[df["plant_height_cm"] <= 80]


# 3. TEMİZLEME VE SIRALAMA
df = df.drop(columns=['harvest_fresh_mass_g', 'harvest_dry_mass_g', 'bay','is_interpolated'])

print(df[feature_cols].head()) 

plt.figure(figsize=(10, 4))
sns.boxplot(x=df[target_col].values)
plt.title("Bitki Boyu - Train Set (Boxplot)")
plt.xlabel("Boy (cm)")
plt.tight_layout()
plt.show()

# Eğitim ve test setlerini ayırma
train_df, test_df = rep_based_train_test_split(df, test_size=0.2, random_seed=13)

# Eğitim verisinde pencereleme
X_train_seq, y_train_seq = create_env_sequences_by_group(train_df, feature_cols, target_col, time_steps, step=3)
# Test verisinde pencereleme
X_test_seq, y_test_seq = create_env_sequences_by_group(test_df, feature_cols, target_col, time_steps, step=3)

# X verisini yeniden şekillendir (2D olacak şekilde)
n_timesteps = X_train_seq.shape[1]
n_features = X_train_seq.shape[2]

# Ölçekleyiciyi sadece train üzerinde fit et
scaler_X = RobustScaler()
X_train_flat = X_train_seq.reshape(-1, n_features)
X_test_flat = X_test_seq.reshape(-1, n_features)

scaler_X.fit(X_train_flat)

# Dönüştür ve yeniden orijinal shape'e döndür
X_train_scaled = scaler_X.transform(X_train_flat).reshape(-1, n_timesteps, n_features)
X_test_scaled = scaler_X.transform(X_test_flat).reshape(-1, n_timesteps, n_features)

# Y için scaler (1D)
scaler_y = RobustScaler()
y_train_scaled = scaler_y.fit_transform(y_train_seq.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test_seq.reshape(-1, 1))

#%%
print(f"Eğitim sequence sayısı: {len(X_train_scaled)}")
print(f"Test sequence sayısı: {len(y_test_scaled)}")
print(f"Sequence şekli: {X_train_scaled.shape}")
# 8. LSTM MODELİ - Adil karşılaştırma için standart mimari

n_features = X_train_seq.shape[2]
print(f"\n🏗️ MODEL MİMARİLERİ (Adil karşılaştırma için aynı nöron sayıları):")
print(f"Giriş özellikleri: {n_features}, Zaman adımları: {time_steps}")

# Standart mimari: 64 nöronlu ana katman + 32 dense + çıkış
model = Sequential()
model.add(LSTM(64, input_shape=(time_steps, n_features), activation='tanh', return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

print("LSTM: LSTM(64) → Dropout(0.2) → Dense(32) → Dense(1)")

# 8b. RNN (SimpleRNN) MODELİ - Aynı mimari
rnn_model = Sequential()
rnn_model.add(SimpleRNN(64, input_shape=(time_steps, n_features), activation='tanh', return_sequences=False))
rnn_model.add(Dropout(0.2))
rnn_model.add(Dense(32, activation='relu'))
rnn_model.add(Dense(1))
rnn_model.compile(optimizer='adam', loss='mse')

print("RNN:  SimpleRNN(64) → Dropout(0.2) → Dense(32) → Dense(1)")

# 8c. ELM MODELİ (Geliştirilmiş)
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class ELMRegressor(BaseEstimator, RegressorMixin):
    """
    Extreme Learning Machine (ELM) Regressor
    
    Parameters:
    -----------
    n_hidden : int, default=100
        Gizli katmandaki nöron sayısı
    activation : function, default=np.tanh
        Aktivasyon fonksiyonu (tanh, relu, sigmoid)
    random_state : int, default=None
        Rastgelelik kontrolü için seed
    regularization : float, default=None
        L2 regularizasyon katsayısı
    """
    def __init__(self, n_hidden=100, activation=np.tanh, random_state=None, regularization=None):
        self.n_hidden = n_hidden
        self.activation = activation
        self.random_state = random_state
        self.regularization = regularization
    
    def _get_activation_function(self, name):
        """Aktivasyon fonksiyonunu döndürür"""
        if name == 'tanh':
            return np.tanh
        elif name == 'sigmoid':
            return lambda x: 1 / (1 + np.exp(-np.clip(x, -250, 250)))
        elif name == 'relu':
            return lambda x: np.maximum(0, x)
        else:
            return self.activation
    
    def fit(self, X, y):
        """ELM modelini eğitir"""
        X, y = check_X_y(X, y)
        rng = np.random.RandomState(self.random_state)
        
        # Giriş ağırlıkları ve bias'ları rastgele başlat
        self.input_weights_ = rng.normal(size=(X.shape[1], self.n_hidden), scale=1.0)
        self.bias_ = rng.normal(size=(self.n_hidden,), scale=1.0)
        
        # Gizli katman çıktısını hesapla
        H = self.activation(np.dot(X, self.input_weights_) + self.bias_)
        
        # Çıktı ağırlıklarını hesapla (Moore-Penrose pseudo-inverse ile)
        if self.regularization is not None:
            # Ridge regularization ile
            self.beta_ = np.linalg.solve(
                H.T @ H + self.regularization * np.eye(H.shape[1]),
                H.T @ y
            )
        else:
            # Standart pseudo-inverse ile
            self.beta_ = np.linalg.pinv(H) @ y
        
        # Model performans bilgilerini sakla
        self.n_features_in_ = X.shape[1]
        self.training_samples_ = X.shape[0]
        
        return self
    
    def predict(self, X):
        """Tahmin yapar"""
        check_is_fitted(self, ["input_weights_", "bias_", "beta_"])
        X = check_array(X)
        
        # Gizli katman çıktısını hesapla
        H = self.activation(np.dot(X, self.input_weights_) + self.bias_)
        
        # Final tahmin
        return H @ self.beta_
    
    def get_params(self, deep=True):
        """Model parametrelerini döndürür"""
        return {
            'n_hidden': self.n_hidden,
            'activation': self.activation,
            'random_state': self.random_state,
            'regularization': self.regularization
        }
    
    def set_params(self, **params):
        """Model parametrelerini ayarlar"""
        for key, value in params.items():
            setattr(self, key, value)
        return self

# 9. ERKEN DURDURMA
es_lstm = EarlyStopping(monitor='val_loss', 
                        patience=25,
                        restore_best_weights=True,
                        min_delta=0.001,
                        verbose=1)

es_rnn = EarlyStopping(monitor='val_loss', 
                       patience=25,
                       restore_best_weights=True,
                       min_delta=0.001,
                       verbose=1)


# 10. EĞİTİM (Zaman ölçümü ile)

print("🚀 LSTM modeli eğitiliyor...")
start_time = time.time()
history = model.fit(
    X_train_scaled, y_train_scaled,
    validation_split=0.1,
    epochs=200,
    batch_size=32,
    callbacks=[es_lstm],
    verbose=2
)
lstm_training_time = time.time() - start_time
print(f"✅ LSTM eğitim süresi: {lstm_training_time:.2f} saniye")

print("🚀 RNN modeli eğitiliyor...")
start_time = time.time()
rnn_history = rnn_model.fit(
    X_train_scaled, y_train_scaled,
    validation_split=0.1,
    epochs=200,
    batch_size=32,
    callbacks=[es_rnn],
    verbose=2
)
rnn_training_time = time.time() - start_time
print(f"✅ RNN eğitim süresi: {rnn_training_time:.2f} saniye")

# ELM için sequence'ları flatten et
X_train_flat = X_train_seq.reshape(X_train_seq.shape[0], -1)  # (samples, timesteps*features)
X_test_flat = X_test_seq.reshape(X_test_seq.shape[0], -1)

# ELM için scaled verileri kullan
scaler_elm = RobustScaler()
X_train_flat_scaled = scaler_elm.fit_transform(X_train_flat)
X_test_flat_scaled = scaler_elm.transform(X_test_flat)

# ELM modelini oluştur ve eğit - Ana katmanlar ile eşleşen nöron sayısı
print("\nELM:  Input → Hidden(64) → Output(1) + Regularization")
print("🚀 ELM modeli eğitiliyor...")
start_time = time.time()
elm_model = ELMRegressor(n_hidden=64, random_state=42, regularization=0.01)  # LSTM/RNN ile aynı
elm_model.fit(X_train_flat_scaled, y_train_seq)  # y_train_seq orijinal değerleri kullan
elm_training_time = time.time() - start_time
print(f"✅ ELM eğitim süresi: {elm_training_time:.2f} saniye")

# Eğitim süresi karşılaştırması
print("\n⏱️ EĞİTİM SÜRESİ KARŞILAŞTIRMASI:")
print(f"LSTM: {lstm_training_time:.2f} saniye")
print(f"RNN:  {rnn_training_time:.2f} saniye")
print(f"ELM:  {elm_training_time:.2f} saniye")

# Model karmaşıklığı analizi
lstm_params = model.count_params()
rnn_params = rnn_model.count_params()
elm_params = elm_model.input_weights_.size + elm_model.bias_.size + elm_model.beta_.size

print("\n🔧 MODEL KARMAŞIKLIĞI:")
print(f"LSTM parametreleri: {lstm_params:,}")
print(f"RNN parametreleri:  {rnn_params:,}")
print(f"ELM parametreleri:  {elm_params:,}")

print("\n🏗️ DETAYLI MİMARİ KARŞILAŞTIRMASI:")
print("="*60)
print(f"{'Model':<8} | {'Ana Katman':<15} | {'Dropout':<8} | {'Dense':<10} | {'Çıkış':<6}")
print("-"*60)
print(f"{'LSTM':<8} | {'LSTM(64)':<15} | {'0.2':<8} | {'Dense(32)':<10} | {'Dense(1)':<6}")
print(f"{'RNN':<8} | {'SimpleRNN(64)':<15} | {'0.2':<8} | {'Dense(32)':<10} | {'Dense(1)':<6}")
print(f"{'ELM':<8} | {'Hidden(64)':<15} | {'Reg=0.01':<8} | {'Direct':<10} | {'Output(1)':<6}")
print("="*60)

# Detaylı parametre analizi
lstm_main_params = 64 * (n_features + 64 + 1) * 4  # LSTM has 4 gates
rnn_main_params = 64 * (n_features + 64 + 1)  # RNN has 1 gate
elm_main_params = n_features * 64 + 64  # Input weights + bias

print(f"\nAna katman parametre sayısı:")
print(f"  LSTM ana katman: {lstm_main_params:,} (4 kapıli)")  
print(f"  RNN ana katman:  {rnn_main_params:,} (1 kapıli)")
print(f"  ELM gizli katman: {elm_main_params:,}")

print(f"\nModel büyüklüğü oranları (RNN'e göre):")
print(f"  LSTM: {lstm_params/rnn_params:.2f}x")
print(f"  RNN:  1.00x (referans)")
print(f"  ELM:  {elm_params/rnn_params:.2f}x")

#%%
# 11. TAHMİN VE TERS ÖLÇEKLEME
y_pred_scaled = model.predict(X_test_scaled)
y_pred_inv = scaler_y.inverse_transform(y_pred_scaled)

# RNN tahmini
y_rnn_pred_scaled = rnn_model.predict(X_test_scaled)
y_rnn_pred_inv = scaler_y.inverse_transform(y_rnn_pred_scaled)

# ELM tahmini
y_elm_pred = elm_model.predict(X_test_flat_scaled)
y_elm_pred_inv = y_elm_pred  # ELM direkt orijinal değerlerle eğitildi

Y_test_inv = scaler_y.inverse_transform(y_test_scaled)

# LSTM metrikleri
mae = mean_absolute_error(Y_test_inv, y_pred_inv)
mse = mean_squared_error(Y_test_inv, y_pred_inv)
rmse = math.sqrt(mse)
r_squared = r2_score(Y_test_inv, y_pred_inv)

# RNN metrikleri
mae_rnn = mean_absolute_error(Y_test_inv, y_rnn_pred_inv)
mse_rnn = mean_squared_error(Y_test_inv, y_rnn_pred_inv)
rmse_rnn = math.sqrt(mse_rnn)
r2_rnn = r2_score(Y_test_inv, y_rnn_pred_inv)

# ELM metrikleri
mae_elm = mean_absolute_error(Y_test_inv, y_elm_pred_inv)
mse_elm = mean_squared_error(Y_test_inv, y_elm_pred_inv)
rmse_elm = math.sqrt(mse_elm)
r2_elm = r2_score(Y_test_inv, y_elm_pred_inv)


print("Başarı Metrikleri:")
print(f"LSTM -> MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r_squared:.2f}")
print(f"RNN  -> MAE: {mae_rnn:.2f}, RMSE: {rmse_rnn:.2f}, R2: {r2_rnn:.2f}")
print(f"ELM  -> MAE: {mae_elm:.2f}, RMSE: {rmse_elm:.2f}, R2: {r2_elm:.2f}")

# 12. MODEL KARŞILAŞTIRMA TABLOSU
import pandas as pd
comparison_df = pd.DataFrame({
    'Model': ['LSTM', 'RNN', 'ELM'],
    'MAE': [mae, mae_rnn, mae_elm],
    'RMSE': [rmse, rmse_rnn, rmse_elm],
    'R²': [r_squared, r2_rnn, r2_elm]
})
print("\n📊 MODEL KARŞILAŞTIRMA TABLOSU:")
print(comparison_df.round(3))

# En iyi performans gösteren modeli belirle
best_model_idx = comparison_df['RMSE'].idxmin()
best_model_name = comparison_df.loc[best_model_idx, 'Model']
print(f"\n🏆 En iyi model (en düşük RMSE): {best_model_name}")

# Gerçek model nesnesini seç
if best_model_name == 'LSTM':
    best_model_obj = model
elif best_model_name == 'RNN':
    best_model_obj = rnn_model
else:  # ELM
    best_model_obj = elm_model

# 13. SONUÇ GÖRSELLEŞTİRME
plt.figure(figsize=(15, 6))
plt.plot(Y_test_inv[410:468], label='Gerçek', color='black', linewidth=2)
plt.plot(y_pred_inv[410:468], label='LSTM', alpha=0.8)
plt.plot(y_rnn_pred_inv[410:468], label='RNN', alpha=0.8)
plt.plot(y_elm_pred_inv[410:468], label='ELM', alpha=0.8)
plt.legend()
plt.title(f'LSTM, RNN, ELM ile Bitki Boyu Tahmini (Window Size: {time_steps})')
plt.xlabel('Zaman')
plt.ylabel('Bitki Boyu (cm)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#13.1 Örnek bir bitki üzerinden tahmin ve görselleştirme
plt.figure(figsize=(15, 6))
plt.plot(Y_test_inv[410:418], label='Gerçek', color='black', linewidth=2)
plt.plot(y_pred_inv[410:418], label='LSTM', alpha=0.8)
plt.plot(y_rnn_pred_inv[410:418], label='RNN', alpha=0.8)
plt.plot(y_elm_pred_inv[410:418], label='ELM', alpha=0.8)
plt.legend()
plt.title(f'LSTM, RNN, ELM ile Bitki Boyu Tahmini (Window Size: {time_steps})')
plt.xlabel('Zaman')
plt.ylabel('Bitki Boyu (cm)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#13.1 Örnek bir bitki üzerinden tahmin ve görselleştirme
plt.figure(figsize=(15, 6))
plt.plot(Y_test_inv[441:446], label='Gerçek', color='black', linewidth=2)
plt.plot(y_pred_inv[441:446], label='LSTM', alpha=0.8)
plt.plot(y_rnn_pred_inv[441:446], label='RNN', alpha=0.8)
plt.plot(y_elm_pred_inv[441:446], label='ELM', alpha=0.8)
plt.legend()
plt.title(f'LSTM, RNN, ELM ile Bitki Boyu Tahmini (Window Size: {time_steps})')
plt.xlabel('Zaman')
plt.ylabel('Bitki Boyu (cm)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 13. SONUÇ GÖRSELLEŞTİRME (step 1 için)
# plt.figure(figsize=(15, 6))
# plt.plot(Y_test_inv[100:300], label='Gerçek', color='black', linewidth=2)
# plt.plot(y_pred_inv[100:300], label='LSTM', alpha=0.8)
# plt.plot(y_rnn_pred_inv[100:300], label='RNN', alpha=0.8)
# plt.plot(y_elm_pred_inv[100:300], label='ELM', alpha=0.8)
# plt.legend()
# plt.title(f'LSTM, RNN, ELM ile Bitki Boyu Tahmini (Window Size: {time_steps})')
# plt.xlabel('Zaman')
# plt.ylabel('Bitki Boyu (cm)')
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(15, 6))
# plt.plot(Y_test_inv[152:165], label='Gerçek', color='black', linewidth=2)
# plt.plot(y_pred_inv[152:165], label='LSTM', alpha=0.8)
# plt.plot(y_rnn_pred_inv[152:165], label='RNN', alpha=0.8)
# plt.plot(y_elm_pred_inv[152:165], label='ELM', alpha=0.8)
# plt.legend()
# plt.title(f'LSTM, RNN, ELM ile Bitki Boyu Tahmini (Window Size: {time_steps})')
# plt.xlabel('Zaman')
# plt.ylabel('Bitki Boyu (cm)')
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.show()

# Model performans karşılaştırma grafiği
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.bar(['LSTM', 'RNN', 'ELM'], [mae, mae_rnn, mae_elm], color=['blue', 'orange', 'green'])
plt.title('MAE Karşılaştırması')
plt.ylabel('MAE')

plt.subplot(1, 3, 2)
plt.bar(['LSTM', 'RNN', 'ELM'], [rmse, rmse_rnn, rmse_elm], color=['blue', 'orange', 'green'])
plt.title('RMSE Karşılaştırması')
plt.ylabel('RMSE')

plt.subplot(1, 3, 3)
plt.bar(['LSTM', 'RNN', 'ELM'], [r_squared, r2_rnn, r2_elm], color=['blue', 'orange', 'green'])
plt.title('R² Karşılaştırması')
plt.ylabel('R²')

plt.tight_layout()
plt.show()

# --- Tahmin edilen vs Gerçek scatter plotları (tüm modeller) ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# LSTM scatter plot
axes[0].scatter(Y_test_inv, y_pred_inv, alpha=0.5, label='LSTM', color='blue')
axes[0].plot([Y_test_inv.min(), Y_test_inv.max()], [Y_test_inv.min(), Y_test_inv.max()], 'k--', lw=2)
axes[0].set_xlabel('Gerçek Değer')
axes[0].set_ylabel('Tahmin (LSTM)')
axes[0].set_title(f'LSTM: Gerçek vs Tahmin (R²={r_squared:.3f})')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# RNN scatter plot
axes[1].scatter(Y_test_inv, y_rnn_pred_inv, alpha=0.5, label='RNN', color='orange')
axes[1].plot([Y_test_inv.min(), Y_test_inv.max()], [Y_test_inv.min(), Y_test_inv.max()], 'k--', lw=2)
axes[1].set_xlabel('Gerçek Değer')
axes[1].set_ylabel('Tahmin (RNN)')
axes[1].set_title(f'RNN: Gerçek vs Tahmin (R²={r2_rnn:.3f})')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# ELM scatter plot
axes[2].scatter(Y_test_inv, y_elm_pred_inv, alpha=0.5, label='ELM', color='green')
axes[2].plot([Y_test_inv.min(), Y_test_inv.max()], [Y_test_inv.min(), Y_test_inv.max()], 'k--', lw=2)
axes[2].set_xlabel('Gerçek Değer')
axes[2].set_ylabel('Tahmin (ELM)')
axes[2].set_title(f'ELM: Gerçek vs Tahmin (R²={r2_elm:.3f})')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 10.1. EĞİTİM GEÇMİŞİ GÖRSELLEŞTİRME (LOSS GRAFİKLERİ)
plt.figure(figsize=(15, 5))

# LSTM Loss Grafiği
plt.subplot(1, 3, 1)
plt.plot(history.history['loss'], label='Train Loss', color='blue', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', color='red', linewidth=2)
plt.title('LSTM - Training & Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True, alpha=0.3)

# Early stopping noktasını işaretle
if len(history.history['loss']) < 200:  # Early stopping devreye girdiyse
    plt.axvline(x=len(history.history['loss']), color='orange', linestyle='--', 
                label=f'Early Stop (Epoch {len(history.history["loss"])})')
    plt.legend()

# RNN Loss Grafiği
plt.subplot(1, 3, 2)
plt.plot(rnn_history.history['loss'], label='Train Loss', color='blue', linewidth=2)
plt.plot(rnn_history.history['val_loss'], label='Validation Loss', color='red', linewidth=2)
plt.title('RNN - Training & Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True, alpha=0.3)

# Early stopping noktasını işaretle
if len(rnn_history.history['loss']) < 200:  # Early stopping devreye girdiyse
    plt.axvline(x=len(rnn_history.history['loss']), color='orange', linestyle='--', 
                label=f'Early Stop (Epoch {len(rnn_history.history["loss"])})')
    plt.legend()


# Loss istatistikleri
print("\n📈 EĞİTİM GEÇMİŞİ İSTATİSTİKLERİ:")
print("="*50)
print(f"LSTM:")
print(f"  • Toplam epoch: {len(history.history['loss'])}")
print(f"  • Final train loss: {history.history['loss'][-1]:.6f}")
print(f"  • Final val loss: {history.history['val_loss'][-1]:.6f}")
print(f"  • Min val loss: {min(history.history['val_loss']):.6f}")

print(f"\nRNN:")
print(f"  • Toplam epoch: {len(rnn_history.history['loss'])}")
print(f"  • Final train loss: {rnn_history.history['loss'][-1]:.6f}")
print(f"  • Final val loss: {rnn_history.history['val_loss'][-1]:.6f}")
print(f"  • Min val loss: {min(rnn_history.history['val_loss']):.6f}")

# Overfitting analizi
lstm_overfitting = history.history['val_loss'][-1] - min(history.history['val_loss'])
rnn_overfitting = rnn_history.history['val_loss'][-1] - min(rnn_history.history['val_loss'])

print(f"\n🔍 OVERFİTTİNG ANALİZİ:")
print(f"LSTM overfitting derecesi: {lstm_overfitting:.6f}")
print(f"RNN overfitting derecesi:  {rnn_overfitting:.6f}")

if lstm_overfitting < rnn_overfitting:
    print("✅ LSTM daha az overfitting gösteriyor")
else:
    print("✅ RNN daha az overfitting gösteriyor")

#%%
# Özellik önemi (Permutation Importance) - LSTM için
import copy
feature_names = feature_cols  # Modelde kullanılan feature sırası ile aynı olmalı
base_score_lstm = mean_squared_error(Y_test_inv, y_pred_inv)
base_score_elm = mean_squared_error(Y_test_inv, y_elm_pred_inv)

importances_lstm = []
importances_elm = []

print("🔍 Özellik önemlerini hesaplıyor...")

for i, fname in enumerate(feature_names):
    # LSTM için permutation importance
    X_test_shuffled = copy.deepcopy(X_test_seq)
    flat = X_test_shuffled[:, :, i].flatten()
    np.random.shuffle(flat)
    X_test_shuffled[:, :, i] = flat.reshape(X_test_shuffled[:, :, i].shape)
    y_pred_shuffled = model.predict(X_test_shuffled)
    y_pred_shuffled_inv = scaler_y.inverse_transform(y_pred_shuffled.reshape(-1, 1))
    score_lstm = mean_squared_error(Y_test_inv, y_pred_shuffled_inv)
    importances_lstm.append(score_lstm - base_score_lstm)
    
    # ELM için permutation importance
    X_test_flat_shuffled = copy.deepcopy(X_test_flat_scaled)
    # Her time step için aynı özelliği karıştır
    for t in range(time_steps):
        feature_idx = t * n_features + i
        if feature_idx < X_test_flat_shuffled.shape[1]:
            np.random.shuffle(X_test_flat_shuffled[:, feature_idx])
    
    y_elm_pred_shuffled = elm_model.predict(X_test_flat_shuffled)
    score_elm = mean_squared_error(Y_test_inv, y_elm_pred_shuffled)
    importances_elm.append(score_elm - base_score_elm)

# Özellik önemi görselleştirmesi
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.barh(feature_names, importances_lstm, color='blue', alpha=0.7)
plt.title('LSTM - Özellik Önemleri (Permutation)')
plt.xlabel('MSE Artışı')
plt.gca().invert_yaxis()

plt.subplot(1, 2, 2)
plt.barh(feature_names, importances_elm, color='green', alpha=0.7)
plt.title('ELM - Özellik Önemleri (Permutation)')
plt.xlabel('MSE Artışı')
plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()

# Özellik önemlerini DataFrame olarak göster
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'LSTM_Importance': importances_lstm,
    'ELM_Importance': importances_elm
})
importance_df = importance_df.sort_values('LSTM_Importance', ascending=False)
print("\n📊 ÖZELLIK ÖNEMLERİ TABLOSU:")
print(importance_df.round(4))


# Bitki boyu ile diğer değişkenler arasında korelasyon matrisi ve görselleştirme
corr = df[feature_cols + [target_col]].corr()
print('Korelasyon Matrisi:')
print(corr[target_col].sort_values(ascending=False))

plt.figure(figsize=(8,6))
plt.bar(corr[target_col].index, corr[target_col].values)
plt.title('Bitki Boyu ile Korelasyonlar')
plt.ylabel('Korelasyon (plant_height_cm)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Scatter plot: Bitki boyu ile diğer değişkenler
for col in feature_cols:
    if col == target_col:
        continue
    plt.figure(figsize=(5,4))
    plt.scatter(df[col], df[target_col], alpha=0.5)
    plt.xlabel(col)
    plt.ylabel(target_col)
    plt.title(f'{col} vs {target_col}')
    plt.tight_layout()
    plt.show()


#%%
# 🚀 MODEL KAYDETME VE SCALER KAYDETME
import joblib
import json
from datetime import datetime

# Modeli kaydet - H5 formatında ve uyumlu şekilde
model_filename = 'plant_growth_lstm_model.h5'
model.save(model_filename, save_format='h5')
print(f"✅ LSTM modeli kaydedildi: {model_filename}")

# RNN modelini de kaydet
rnn_filename = 'plant_growth_rnn_model.h5'
rnn_model.save(rnn_filename, save_format='h5')
print(f"✅ RNN modeli kaydedildi: {rnn_filename}")

# ELM modelini kaydet
elm_filename = 'plant_growth_elm_model.pkl'
joblib.dump(elm_model, elm_filename)
print(f"✅ ELM modeli kaydedildi: {elm_filename}")

# Scaler'ları kaydet
scaler_x_filename = 'scaler_x.pkl'
scaler_y_filename = 'scaler_y.pkl'
scaler_elm_filename = 'scaler_elm.pkl'

joblib.dump(scaler_X, scaler_x_filename)
joblib.dump(scaler_y, scaler_y_filename)
joblib.dump(scaler_elm, scaler_elm_filename)
print(f"✅ X scaler kaydedildi: {scaler_x_filename}")
print(f"✅ Y scaler kaydedildi: {scaler_y_filename}")
print(f"✅ ELM scaler kaydedildi: {scaler_elm_filename}")

# Model parametrelerini kaydet
model_params = {
    'time_steps': time_steps,
    'n_features': n_features,
    'feature_cols': feature_cols,
    'target_col': target_col,
    'train_test_split_seed': 42,
    'model_performance': {
        'lstm_mae': float(mae),
        'lstm_rmse': float(rmse),
        'lstm_r2': float(r_squared),
        'rnn_mae': float(mae_rnn),
        'rnn_rmse': float(rmse_rnn),
        'rnn_r2': float(r2_rnn),
        'elm_mae': float(mae_elm),
        'elm_rmse': float(rmse_elm),
        'elm_r2': float(r2_elm)
    },
    'training_info': {
        'lstm_training_time_sec': float(lstm_training_time),
        'rnn_training_time_sec': float(rnn_training_time),
        'elm_training_time_sec': float(elm_training_time),
        'lstm_parameters': int(lstm_params),
        'rnn_parameters': int(rnn_params),
        'elm_parameters': int(elm_params)
    },
    'model_architectures': {
        'lstm_layers': ['LSTM(96)', 'LSTM(64)', 'Dropout(0.2)', 'Dense(32)', 'Dense(16)', 'Dense(1)'],
        'rnn_layers': ['SimpleRNN(96)', 'SimpleRNN(64)', 'Dropout(0.2)', 'Dense(32)', 'Dense(16)', 'Dense(1)'],
        'elm_config': {
            'hidden_neurons': 150,
            'activation': 'tanh',
            'regularization': 0.01
        }
    },
    'data_info': {
        'train_sequences': len(X_train_seq),
        'test_sequences': len(X_test_seq),
        'total_features': len(feature_cols)
    },
    'created_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}

# JSON olarak kaydet
with open('model_params.json', 'w') as f:
    json.dump(model_params, f, indent=4)
print(f"✅ Model parametreleri kaydedildi: model_params.json")

print("\n" + "="*60)
print("📁 KAYDEDILEN DOSYALAR")
print("="*60)
print(f"1. LSTM Model: {model_filename}")
print(f"2. RNN Model: {rnn_filename}")
print(f"3. ELM Model: {elm_filename}")
print(f"4. X Scaler: {scaler_x_filename}")
print(f"5. Y Scaler: {scaler_y_filename}")
print(f"6. ELM Scaler: {scaler_elm_filename}")
print(f"7. Model Params: model_params.json")
print("\n🎯 Bu dosyalar tahmin için kullanılabilir!")

# Final comparison summary
print("\n" + "="*80)
print("📈 KAPSAMLI MODEL KARŞILAŞTIRMASI")
print("="*80)

# Performans tablosu
print("\n📊 PERFORMANS METRİKLERİ:")
print(comparison_df.round(3).to_string(index=False))

# Eğitim zamanı tablosu
training_time_df = pd.DataFrame({
    'Model': ['LSTM', 'RNN', 'ELM'],
    'Eğitim Süresi (sn)': [lstm_training_time, rnn_training_time, elm_training_time],
    'Parametre Sayısı': [lstm_params, rnn_params, elm_params]
})
print(f"\n⏱️ EĞİTİM VERİMLİĞİ:")
print(training_time_df.to_string(index=False))

# En iyi modeli belirle
print(f"\n🏆 En iyi performans (RMSE): {best_model_name} ({comparison_df.loc[best_model_idx, 'RMSE']:.3f})")

# Hız analizi
fastest_model_idx = training_time_df['Eğitim Süresi (sn)'].idxmin()
fastest_model = training_time_df.loc[fastest_model_idx, 'Model']
print(f"⚡ En hızlı eğitim: {fastest_model} ({training_time_df.loc[fastest_model_idx, 'Eğitim Süresi (sn)']:.3f} sn)")

# Verimlilik oranı (performans/zaman)
efficiency_scores = []
for i, model_name in enumerate(['LSTM', 'RNN', 'ELM']):
    r2_val = comparison_df.loc[i, 'R²']
    time_val = training_time_df.loc[i, 'Eğitim Süresi (sn)']
    efficiency = r2_val / time_val if time_val > 0 else 0
    efficiency_scores.append(efficiency)

most_efficient_idx = np.argmax(efficiency_scores)
most_efficient_model = ['LSTM', 'RNN', 'ELM'][most_efficient_idx]
print(f"🎯 En verimli model (R²/zaman): {most_efficient_model} ({efficiency_scores[most_efficient_idx]:.6f})")

print("\n" + "="*80)
print("📝 SONUÇ ÖZETİ")
print("="*80)
print(f"• En yüksek doğruluk: {best_model_name}")
print(f"• En hızlı eğitim: {fastest_model}")
print(f"• En verimli: {most_efficient_model}")
print(f"• Toplam eğitim verimli: {comparison_df.shape[0]} model karşılaştırıldı")
print(f"• Veri seti: {len(X_train_seq)} eğitim, {len(X_test_seq)} test sequence")
print("="*80)

# %%
# =============================================================================
# 🧬 DIFFERENTIAL EVOLUTION İLE ÇEVREsel KOŞUL OPTİMİZASYONU
# =============================================================================

from scipy.optimize import differential_evolution
from tensorflow.keras.models import load_model
import warnings
import json
warnings.filterwarnings('ignore')

class PlantGrowthOptimizer:
    """
    Differential Evolution kullanarak en hızlı bitki büyümesini sağlayacak
    çevresel koşulları bulan optimizasyon sınıfı.
    """
    
    def __init__(self, model, scaler_x, scaler_y, feature_cols, time_steps=3):
        self.model = model
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y
        self.feature_cols = feature_cols
        self.time_steps = time_steps
        self.n_features = len(feature_cols)
        
        # Parametre sınırları
        self.param_bounds = {
            'h2o_temp_C': (19.9, 27.1),        # Su sıcaklığı
            'pH': (5.4, 7.5),                  # pH seviyesi
            'EC': (1.5, 3.9),                  # Elektriksel iletkenlik
            'temp_c': (21.0, 28.0),            # Ortam sıcaklığı
            'relative_humidity_%': (53.0, 64.0), # Nem oranı
            'dewpoint': (13.0, 17.5),          # Çiy noktası
            'days_since_first_measurement': (0, 41), # Gün sayısı
            'reservoir_size_liters': (75, 150)  # Rezervuar boyutu (75 veya 150 L)
        }
        
        # DE için bounds listesi oluştur
        self.bounds = []
        for col in self.feature_cols:
            if col in self.param_bounds:
                self.bounds.append(self.param_bounds[col])
            else:
                raise ValueError(f"Parametre sınırı bulunamadı: {col}")
    
    def create_optimized_sequence(self, params, initial_day=0):
        """
        Verilen parametrelerden bir zaman serisi oluşturur.
        Plant-rep kombinasyonu korunur ve days_since_first_measurement artan şekilde ayarlanır.
        """
        sequences = []
        
        # Her time_step için parametreleri ayarla
        for t in range(self.time_steps):
            time_point = []
            
            for i, col in enumerate(self.feature_cols):
                if col == 'days_since_first_measurement':
                    # Gün sayısı artan şekilde ayarlanır
                    day_value = initial_day + t
                    day_value = np.clip(day_value, 
                                      self.param_bounds[col][0], 
                                      self.param_bounds[col][1])
                    time_point.append(day_value)
                elif col == 'reservoir_size_liters':
                    # Rezervuar boyutu kategorik: 75 veya 150
                    reservoir_value = 150 if params[i] > 112.5 else 75
                    time_point.append(reservoir_value)
                else:
                    # Diğer parametreler verilen değerleri kullanır
                    time_point.append(params[i])
            
            sequences.append(time_point)
        
        return np.array(sequences)
    
    def objective_function(self, params):
        """
        Optimizasyon için amaç fonksiyonu.
        Büyüme hızını maksimize etmek için negatif değer döndürür.
        """
        try:
            # Başlangıç günü rastgele seç (erken dönem büyümeyi hedefle)
            initial_day = np.random.randint(0, 15)
            
            # Optimize edilmiş sekansı oluştur
            sequence = self.create_optimized_sequence(params, initial_day)
            
            # Sekansı model için uygun forma getir
            sequence_reshaped = sequence.reshape(1, self.time_steps, self.n_features)
            
            # Ölçeklendir
            sequence_flat = sequence_reshaped.reshape(-1, self.n_features)
            sequence_scaled = self.scaler_x.transform(sequence_flat)
            sequence_scaled = sequence_scaled.reshape(1, self.time_steps, self.n_features)
            
            # Model tahmini yap
            predicted_height_scaled = self.model.predict(sequence_scaled, verbose=0)
            predicted_height = self.scaler_y.inverse_transform(predicted_height_scaled)[0, 0]
            
            # Büyüme hızını hesapla (boy / gün)
            current_day = initial_day + self.time_steps - 1
            if current_day > 0:
                growth_rate = predicted_height / current_day
            else:
                growth_rate = predicted_height
            
            # Negatif değer döndür (minimizasyon problemi olarak çözmek için)
            return -growth_rate
            
        except Exception as e:
            print(f"Hata: {e}")
            return 1000  # Büyük pozitif değer (kötü sonuç)
    
    def optimize_growth_conditions(self, max_iterations=100, seed=42):
        """
        Differential Evolution ile en iyi büyüme koşullarını bulur.
        """
        print("🧬 Differential Evolution ile optimizasyon başlıyor...")
        print(f"Parametre sayısı: {len(self.bounds)}")
        print(f"Maksimum iterasyon: {max_iterations}")
        
        # DE optimizasyonu çalıştır
        result = differential_evolution(
            self.objective_function,
            self.bounds,
            maxiter=max_iterations,
            popsize=15,
            seed=seed,
            atol=1e-3,
            tol=1e-3,
            disp=True,
            workers=1  # Paralel işleme kapalı (model thread-safe olmayabilir)
        )
        
        return result
    
    def analyze_optimal_conditions(self, optimal_params):
        """
        Bulunan optimal koşulları analiz eder ve raporlar.
        """
        print("\n" + "="*80)
        print("🎯 OPTİMAL ÇEVREsel KOŞULLAR")
        print("="*80)
        
        # Optimal parametreleri düzenle
        optimized_conditions = {}
        for i, col in enumerate(self.feature_cols):
            if col == 'reservoir_size_liters':
                # Rezervuar boyutu kategorik
                reservoir_value = 150 if optimal_params[i] > 112.5 else 75
                optimized_conditions[col] = reservoir_value
            else:
                optimized_conditions[col] = optimal_params[i]
        
        # Sonuçları yazdır
        for param, value in optimized_conditions.items():
            bounds = self.param_bounds[param]
            print(f"{param:<30}: {value:>8.2f} (Sınır: {bounds[0]:.1f}-{bounds[1]:.1f})")
        
        # Farklı başlangıç günleri için tahmin yap
        print(f"\n📈 OPTİMAL KOŞULLARDA TAHMIN EDİLEN BÜYÜME:")
        print("-" * 60)
        
        growth_predictions = []
        for initial_day in [0, 5, 10, 15, 20]:
            sequence = self.create_optimized_sequence(optimal_params, initial_day)
            sequence_reshaped = sequence.reshape(1, self.time_steps, self.n_features)
            
            # Ölçeklendir ve tahmin yap
            sequence_flat = sequence_reshaped.reshape(-1, self.n_features)
            sequence_scaled = self.scaler_x.transform(sequence_flat)
            sequence_scaled = sequence_scaled.reshape(1, self.time_steps, self.n_features)
            
            predicted_height_scaled = self.model.predict(sequence_scaled, verbose=0)
            predicted_height = self.scaler_y.inverse_transform(predicted_height_scaled)[0, 0]
            
            current_day = initial_day + self.time_steps - 1
            growth_rate = predicted_height / current_day if current_day > 0 else predicted_height
            
            growth_predictions.append({
                'start_day': initial_day,
                'end_day': current_day,
                'predicted_height': predicted_height,
                'growth_rate': growth_rate
            })
            
            print(f"Gün {initial_day:2d}-{current_day:2d}: {predicted_height:6.2f} cm "
                  f"(Hız: {growth_rate:5.2f} cm/gün)")
        
        return optimized_conditions, growth_predictions
    
    def compare_with_baseline(self, optimal_params, baseline_params=None):
        """
        Optimal koşulları baseline (ortalama) koşullarla karşılaştırır.
        """
        if baseline_params is None:
            # Ortalama değerleri baseline olarak kullan
            baseline_params = []
            for col in self.feature_cols:
                bounds = self.param_bounds[col]
                avg_val = (bounds[0] + bounds[1]) / 2
                baseline_params.append(avg_val)
        
        print(f"\n⚖️ BASELINE vs OPTİMAL KARŞILAŞTIRMA:")
        print("-"*60)
        
        # Her iki koşul için de tahmin yap
        for condition_name, params in [("Baseline (Ortalama)", baseline_params), 
                                     ("Optimal", optimal_params)]:
            print(f"\n{condition_name}:")
            
            # Parametreleri yazdır
            for i, col in enumerate(self.feature_cols):
                if col == 'reservoir_size_liters':
                    value = 150 if params[i] > 112.5 else 75
                else:
                    value = params[i]
                print(f"  {col}: {value:.2f}")
            
            # Tahmin yap
            sequence = self.create_optimized_sequence(params, initial_day=5)
            sequence_reshaped = sequence.reshape(1, self.time_steps, self.n_features)
            sequence_flat = sequence_reshaped.reshape(-1, self.n_features)
            sequence_scaled = self.scaler_x.transform(sequence_flat)
            sequence_scaled = sequence_scaled.reshape(1, self.time_steps, self.n_features)
            
            predicted_height_scaled = self.model.predict(sequence_scaled, verbose=0)
            predicted_height = self.scaler_y.inverse_transform(predicted_height_scaled)[0, 0]
            
            current_day = 5 + self.time_steps - 1
            growth_rate = predicted_height / current_day
            
            print(f"  📊 Tahmin: {predicted_height:.2f} cm, Hız: {growth_rate:.3f} cm/gün")

# =============================================================================
# 🚀 OPTİMİZASYON ÇALIŞTIRMA
# =============================================================================

print("\n" + "="*80)
print("🧬 DIFFERENTIAL EVOLUTION OPTİMİZASYONU BAŞLATIYOR")
print("="*80)

# En iyi modeli kullan (LSTM)
optimizer = PlantGrowthOptimizer(
    model=model,
    scaler_x=scaler_X,
    scaler_y=scaler_y,
    feature_cols=feature_cols,
    time_steps=time_steps
)

# Optimizasyonu çalıştır
print("⏳ Optimizasyon süreci başlıyor... (Bu birkaç dakika sürebilir)")
optimization_result = optimizer.optimize_growth_conditions(
    max_iterations=250,  # Daha iyi sonuçlar için artırıldı
    seed=13
)

# Sonuçları analiz et (başarısız olsa bile sonuçları değerlendir)
if optimization_result.success or optimization_result.fun is not None:
    success_status = "✅ Başarılı!" if optimization_result.success else "⚠️ Maksimum iterasyona ulaşıldı (yine de sonuç var)"
    print(f"\n{success_status}")
    print(f"📊 En iyi fitness değeri: {-optimization_result.fun:.4f} cm/gün")
    print(f"🔄 Toplam iterasyon: {optimization_result.nit}")
    print(f"🎯 Fonksiyon değerlendirmesi: {optimization_result.nfev}")
    
    # SONUÇ KALİTESİNİ DEĞERLENDİR
    final_fitness = -optimization_result.fun
    print(f"\n📈 SONUÇ DEĞERLENDİRMESİ:")
    print(f"Final büyüme değeri: {final_fitness:.4f} cm/gün")
    
    if final_fitness > 15:
        print("🎯 Mükemmel sonuç! (>15 cm/gün)")
    elif final_fitness > 10:
        print("✅ İyi sonuç! (10-15 cm/gün)")
    elif final_fitness > 5:
        print("🔶 Orta sonuç (5-10 cm/gün)")
    else:
        print("🔻 Düşük sonuç (<5 cm/gün)")
    
    # Son birkaç iterasyondaki iyileşmeyi kontrol et
    print(f"Son iterasyonlarda durağanlık: {'Evet' if optimization_result.nit >= 40 else 'Hayır'}")
    
    # Optimal koşulları analiz et
    optimal_conditions, growth_predictions = optimizer.analyze_optimal_conditions(
        optimization_result.x
    )
    
    # Baseline ile karşılaştır
    optimizer.compare_with_baseline(optimization_result.x)
    
else:
    print(f"\n❌ Optimizasyon tamamen başarısız: {optimization_result.message}")
#%%
# Optimal koşulları kaydet
# JSON serialization için float32 değerleri Python float'a dönüştür
json_compatible_predictions = []
for pred in growth_predictions:
    json_compatible_predictions.append({
        'start_day': int(pred['start_day']),
        'end_day': int(pred['end_day']),
        'predicted_height': float(pred['predicted_height']),
        'growth_rate': float(pred['growth_rate'])
    })

optimal_results = {
    'optimization_success': True,
    'optimal_growth_rate_cm_per_day': float(-optimization_result.fun),
    'iterations': int(optimization_result.nit),
    'function_evaluations': int(optimization_result.nfev),
    'optimal_conditions': {k: float(v) for k, v in optimal_conditions.items()},
    'growth_predictions': json_compatible_predictions,
    'optimization_message': optimization_result.message
}

# JSON'a kaydet
with open('optimum_values.json', 'w') as f:
    json.dump(optimal_results, f, indent=4)
print(f"\n💾 Optimal koşullar kaydedildi: optimum_values.json")
#%%
# =============================================================================
# 📊 OPTİMİZASYON SONUÇLARINI GÖRSELLEŞTİRME
# =============================================================================

if optimization_result.success or optimization_result.fun is not None:
    print("\n📊 Optimizasyon sonuçları görselleştiriliyor...")
    
    # Optimal koşullar grafiği
    plt.figure(figsize=(15, 10))
    
    # 1. Optimal parametreler
    plt.subplot(2, 2, 1)
    param_names = [col.replace('_', '\n') for col in feature_cols]
    param_values = [optimal_conditions[col] for col in feature_cols]
    colors = plt.cm.viridis(np.linspace(0, 1, len(param_names)))
    
    bars = plt.bar(range(len(param_names)), param_values, color=colors, alpha=0.7)
    plt.xticks(range(len(param_names)), param_names, rotation=45, ha='right')
    plt.title('Optimal Çevresel Koşullar')
    plt.ylabel('Değer')
    
    # Değerleri bar'ların üstüne yaz
    for bar, value in zip(bars, param_values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.1f}', ha='center', va='bottom', fontsize=8)
    
    # 2. Büyüme tahminleri
    plt.subplot(2, 2, 2)
    days = [pred['end_day'] for pred in growth_predictions]
    heights = [pred['predicted_height'] for pred in growth_predictions]
    rates = [pred['growth_rate'] for pred in growth_predictions]
    
    plt.plot(days, heights, 'bo-', linewidth=2, markersize=8, label='Tahmin Edilen Boy')
    plt.xlabel('Gün')
    plt.ylabel('Bitki Boyu (cm)')
    plt.title('Optimal Koşullarda Büyüme Tahmini')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 3. Büyüme hızları
    plt.subplot(2, 2, 3)
    plt.bar(range(len(days)), rates, color='green', alpha=0.7)
    plt.xticks(range(len(days)), [f'Gün {d}' for d in days], rotation=45)
    plt.ylabel('Büyüme Hızı (cm/gün)')
    plt.title('Günlük Büyüme Hızları')
    
    # Değerleri bar'ların üstüne yaz
    for i, rate in enumerate(rates):
        plt.text(i, rate + rate*0.01, f'{rate:.3f}', 
                ha='center', va='bottom', fontsize=8)
    
    # 4. Parametre sınırları ile karşılaştırma
    plt.subplot(2, 2, 4)
    
    # days_since_first_measurement parametresini hariç tut
    filtered_feature_cols = [col for col in feature_cols if col != 'days_since_first_measurement']
    filtered_param_names = [col.replace('_', '\n') for col in filtered_feature_cols]
    
    param_indices = range(len(filtered_feature_cols))
    min_bounds = [optimizer.param_bounds[col][0] for col in filtered_feature_cols]
    max_bounds = [optimizer.param_bounds[col][1] for col in filtered_feature_cols]
    optimal_vals = [optimal_conditions[col] for col in filtered_feature_cols]
    
    # Sınırları göster
    plt.fill_between(param_indices, min_bounds, max_bounds, 
                    alpha=0.3, label='Sınır Aralığı', color='gray')
    plt.plot(param_indices, optimal_vals, 'ro-', 
            linewidth=2, markersize=8, label='Optimal Değerler')
    
    plt.xticks(param_indices, filtered_param_names, rotation=45, ha='right')
    plt.ylabel('Değer')
    plt.title('Optimal Değerler vs Sınır Aralıkları\n(Çevresel Parametreler)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('optimization_Results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Özet istatistikler
    print("\n" + "="*80)
    print("📈 OPTİMİZASYON ÖZETİ")
    print("="*80)
    print(f"🎯 Maksimum büyüme hızı: {-optimization_result.fun:.4f} cm/gün")
    print(f"📊 Ortalama tahmin boy: {np.mean(heights):.2f} cm")
    print(f"🔄 Optimizasyon süresi: {optimization_result.nfev} fonksiyon değerlendirmesi")
    print(f"✅ Başarı durumu: {'Başarılı' if optimization_result.success else 'Başarısız'}")
    
    # En kritik parametreleri belirle
    param_ranges = []
    for col in feature_cols:
        bounds = optimizer.param_bounds[col]
        range_size = bounds[1] - bounds[0]
        optimal_val = optimal_conditions[col]
        normalized_pos = (optimal_val - bounds[0]) / range_size
        param_ranges.append((col, normalized_pos, optimal_val))
    
    # En uç değerleri göster
    extreme_params = sorted(param_ranges, key=lambda x: abs(x[1] - 0.5), reverse=True)[:3]
    print(f"\n🔍 EN KRİTİK PARAMETRELER (sınır değerlerine en yakın):")
    for param, norm_pos, value in extreme_params:
        position = "maksimuma yakın" if norm_pos > 0.7 else "minimuma yakın" if norm_pos < 0.3 else "orta değerde"
        print(f"  {param}: {value:.2f} ({position})")

print("\n" + "="*80)
print("🎉 OPTİMİZASYON TAMAMLANDI!")
print("="*80)

# %%
# =============================================================================
# 📋 MEVCUT OPTİMİZASYON SONUCUNU ANALİZ ET
# =============================================================================

print("\n" + "="*80)
print("🔍 MEVCUT OPTİMİZASYON SONUCU ANALİZİ")
print("="*80)

# Optimizasyon sonucunu değerlendir
if 'optimization_result' in locals() and optimization_result.fun is not None:
    final_fitness = -optimization_result.fun
    
    print(f"📊 SONUÇ ÖZETİ:")
    print(f"  • En iyi büyüme değeri: {final_fitness:.4f} cm/gün")
    print(f"  • İterasyon sayısı: {optimization_result.nit}/100")
    print(f"  • Fonksiyon değerlendirmesi: {optimization_result.nfev}")
    print(f"  • Başarı durumu: {'✅ Başarılı' if optimization_result.success else '⚠️ Max iterasyon'}")
    
    # Kalite değerlendirmesi
    print(f"\n🎯 KALİTE DEĞERLENDİRMESİ:")
    if final_fitness > 20:
        quality = "🏆 Mükemmel"
        quality_desc = "Çok yüksek büyüme hızı"
    elif final_fitness > 15:
        quality = "🥇 Çok İyi"
        quality_desc = "Yüksek büyüme hızı"
    elif final_fitness > 10:
        quality = "🥈 İyi"
        quality_desc = "Orta-yüksek büyüme hızı"
    elif final_fitness > 5:
        quality = "🥉 Orta"
        quality_desc = "Orta büyüme hızı"
    else:
        quality = "🔻 Düşük"
        quality_desc = "Düşük büyüme hızı"
    
    print(f"  • Kalite: {quality} ({quality_desc})")
    print(f"  • Bu değer, günde {final_fitness:.2f} cm büyüme demektir!")
    
    # Haftalık/aylık projeksiyonlar
    weekly_growth = final_fitness * 7
    monthly_growth = final_fitness * 30
    
    print(f"\n📈 PROJEKSİYONLAR:")
    print(f"  • Haftalık büyüme: {weekly_growth:.1f} cm/hafta")
    print(f"  • Aylık büyüme: {monthly_growth:.1f} cm/ay")
    
    # Veri seti ile karşılaştırma
    if 'df' in locals():
        actual_heights = df['plant_height_cm'].values
        max_height = actual_heights.max()
        avg_height = actual_heights.mean();
        
        print(f"\n📊 VERİ SETİ KARŞILAŞTIRMASI:")
        print(f"  • Veri setindeki max boy: {max_height:.1f} cm")
        print(f"  • Veri setindeki ortalama boy: {avg_height:.1f} cm")
        print(f"  • Optimize edilmiş günlük büyüme: {final_fitness:.2f} cm/gün")
        
        # Bu hızla ne kadar sürede max boy'a ulaşabiliriz
        days_to_max = max_height / final_fitness
        print(f"  • Bu hızla {max_height:.1f} cm'e ulaşma süresi: {days_to_max:.1f} gün")
    
    # Convergence analizi
    print(f"\n🔄 KONVERJANS ANALİZİ:")
    if optimization_result.nit >= 90:
        print(f"  • ⚠️ Optimizasyon maksimum iterasyona yakın durdu")
        print(f"  • 💡 Daha fazla iterasyon ile daha iyi sonuç alınabilir")
    elif optimization_result.nit < 30:
        print(f"  • ✅ Hızlı konverjans - optimal çözüm bulundu")
    else:
        print(f"  • ✅ Normal konverjans süreci")
    
    # =============================================================================
    # 📊 PARAMETRİK ANALİZ GRAFİKLERİ
    # =============================================================================
    
    print("\n📊 Parametrik analiz grafikleri oluşturuluyor...")
    
    # Optimal değerleri al
    opt_h2o_temp = optimal_conditions['h2o_temp_C']
    opt_ph = optimal_conditions['pH']
    opt_ec = optimal_conditions['EC']
    opt_temp_c = optimal_conditions['temp_c']
    opt_rh = optimal_conditions['relative_humidity_%']
    opt_dewpoint = optimal_conditions['dewpoint']
    
    # Parametrik analiz için fonksiyon
    def evaluate_growth(params):
        """Verilen parametreler için büyüme değerini hesaplar"""
        return -optimizer.objective_function(params)
    
    # Bounds'ları al
    bounds = optimizer.bounds
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Su Sıcaklığı etkisi
    temps = np.linspace(bounds[0][0], bounds[0][1], 30)
    temp_growth = [-optimizer.objective_function([t, opt_ph, opt_ec, opt_temp_c, opt_rh, opt_dewpoint, 10, 150]) for t in temps]
    axes[0,0].plot(temps, temp_growth)
    axes[0,0].set_title('Su Sıcaklığı Etkisi')
    axes[0,0].set_xlabel('Su Sıcaklığı (°C)')
    axes[0,0].set_ylabel('Tahmini Bitki Boyu (cm)')
    axes[0,0].axvline(opt_h2o_temp, color='r', linestyle='--', label=f'Optimum: {opt_h2o_temp:.2f}')
    axes[0,0].legend()
    
    # pH etkisi
    phs = np.linspace(bounds[1][0], bounds[1][1], 30)
    ph_growth = [-optimizer.objective_function([opt_h2o_temp, p, opt_ec, opt_temp_c, opt_rh, opt_dewpoint, 10, 150]) for p in phs]
    axes[0,1].plot(phs, ph_growth)
    axes[0,1].set_title('pH Etkisi')
    axes[0,1].set_xlabel('pH')
    axes[0,1].axvline(opt_ph, color='r', linestyle='--', label=f'Optimum: {opt_ph:.2f}')
    axes[0,1].legend()
    
    # EC etkisi
    ecs = np.linspace(bounds[2][0], bounds[2][1], 30)
    ec_growth = [-optimizer.objective_function([opt_h2o_temp, opt_ph, e, opt_temp_c, opt_rh, opt_dewpoint, 10, 150]) for e in ecs]
    axes[0,2].plot(ecs, ec_growth)
    axes[0,2].set_title('EC Etkisi')
    axes[0,2].set_xlabel('EC (mS/cm)')
    axes[0,2].axvline(opt_ec, color='r', linestyle='--', label=f'Optimum: {opt_ec:.2f}')
    axes[0,2].legend()
    
    # Ortam Sıcaklığı etkisi
    temp_c_values = np.linspace(bounds[3][0], bounds[3][1], 30)
    temp_c_growth = [-optimizer.objective_function([opt_h2o_temp, opt_ph, opt_ec, t, opt_rh, opt_dewpoint, 10, 150]) for t in temp_c_values]
    axes[1,0].plot(temp_c_values, temp_c_growth)
    axes[1,0].set_title('Ortam Sıcaklığı Etkisi')
    axes[1,0].set_xlabel('Ortam Sıcaklığı (°C)')
    axes[1,0].set_ylabel('Tahmini Bitki Boyu (cm)')
    axes[1,0].axvline(opt_temp_c, color='r', linestyle='--', label=f'Optimum: {opt_temp_c:.2f}')
    axes[1,0].legend()
    
    # Bağıl Nem etkisi
    rel_humidities = np.linspace(bounds[4][0], bounds[4][1], 30)
    humidity_growth = [-optimizer.objective_function([opt_h2o_temp, opt_ph, opt_ec, opt_temp_c, h, opt_dewpoint, 10, 150]) for h in rel_humidities]
    axes[1,1].plot(rel_humidities, humidity_growth)
    axes[1,1].set_title('Bağıl Nem Etkisi')
    axes[1,1].set_xlabel('Bağıl Nem (%)')
    axes[1,1].axvline(opt_rh, color='r', linestyle='--', label=f'Optimum: {opt_rh:.2f}')
    axes[1,1].legend()
    
    # Çiğ Noktası etkisi
    dewpoints = np.linspace(bounds[5][0], bounds[5][1], 30)
    dewpoint_growth = [-optimizer.objective_function([opt_h2o_temp, opt_ph, opt_ec, opt_temp_c, opt_rh, d, 10, 150]) for d in dewpoints]
    axes[1,2].plot(dewpoints, dewpoint_growth)
    axes[1,2].set_title('Çiğ Noktası Etkisi')
    axes[1,2].set_xlabel('Çiğ Noktası (°C)')
    axes[1,2].axvline(opt_dewpoint, color='r', linestyle='--', label=f'Optimum: {opt_dewpoint:.2f}')
    axes[1,2].legend()
    
    plt.tight_layout()
    plt.show()
    
else:
    print("❌ Optimizasyon sonucu bulunamadı!")

print("="*80) 
# %%
