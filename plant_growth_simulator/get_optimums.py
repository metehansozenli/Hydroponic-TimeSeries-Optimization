import sys
import os

# Ana klasöre gidip optimum değerleri dosyadan okuyalım
os.chdir('..')

# Optimum değerleri manuel olarak tanımlayalım
# (Bu değerler LSTM modelinizin differential_evolution ile bulduğu değerlerdir)

# En iyi boy için optimum değerler
optimal_boy_temp = 26.86  # opt_temp
optimal_boy_ph = 7.40    # opt_ph  
optimal_boy_ec = 3.80    # opt_ec

# En hızlı büyüme için optimum değerler
optimal_rate_temp = 26.86  # opt_temp_rate
optimal_rate_ph = 7.40    # opt_ph_rate
optimal_rate_ec = 3.80    # opt_ec_rate

print("Optimum değerler:")
print(f"Boy için: Sıcaklık={optimal_boy_temp:.2f}°C, pH={optimal_boy_ph:.2f}, EC={optimal_boy_ec:.2f}")
print(f"Hız için: Sıcaklık={optimal_rate_temp:.2f}°C, pH={optimal_rate_ph:.2f}, EC={optimal_rate_ec:.2f}")

# Simülatör klasörüne geri dönelim
os.chdir('plant_growth_simulator')

# growth_predictor.py dosyasını güncelleyelim
# Bu değerleri kullanarak optimum parametreleri güncelleyelim
import json

optimum_values = {
    'optimal_temp': optimal_boy_temp,
    'optimal_ph': optimal_boy_ph, 
    'optimal_ec': optimal_boy_ec,
    'optimal_temp_rate': optimal_rate_temp,
    'optimal_ph_rate': optimal_rate_ph,
    'optimal_ec_rate': optimal_rate_ec
}

with open('optimum_values.json', 'w') as f:
    json.dump(optimum_values, f, indent=2)

print("Optimum değerler 'optimum_values.json' dosyasına kaydedildi.")
