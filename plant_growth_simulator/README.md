# Bitki Büyüme Simülatörü

Bu uygulama, belirli çevresel koşullar altında bitki büyümesini tahmin eden interaktif bir araçtır. LSTM modeli kullanarak, su sıcaklığı, pH ve EC değerlerine bağlı olarak bitki büyüme performansını simüle eder.

## Özellikler

- Su sıcaklığı, pH ve EC değerlerini interaktif olarak ayarlama
- Bitki büyümesinin görsel animasyonu
- Büyüme grafiği ve istatistikler
- Optimum koşullarla karşılaştırma
- Gerçek zamanlı ortam değerlendirmesi

## Kurulum

1. Gerekli paketleri yükleyin:
   ```
   pip install flask tensorflow pandas numpy scikit-learn matplotlib
   ```

2. LSTM modelini ve scaler değerlerini hazırlayın:
   ```
   python save_scalers.py
   ```

3. Uygulamayı çalıştırın:
   ```
   python run_plant_simulator.py
   ```

4. Tarayıcınızda http://localhost:5001 adresini açın

## Kullanım

1. Sol paneldeki sürgüleri kullanarak su sıcaklığı, pH ve EC değerlerini ayarlayın
2. "Simülasyonu Çalıştır" düğmesine tıklayın
3. Sağ panelde bitki büyüme tahminlerini ve animasyonu gözlemleyin
4. Ortam değerlendirmesi panelinde seçtiğiniz koşulların kalitesini kontrol edin

## Teknik Detaylar

- LSTM modeli: lstm_plant_growth_model_optimum_env.py dosyasından alınmıştır
- Optimal değerler deneysel verilerden elde edilmiştir
- Web arayüzü: Flask, Bootstrap 5, GSAP animasyon kütüphanesi

## Notlar

Bu simülatör eğitim amaçlıdır ve gerçek bitki büyüme koşullarından farklılık gösterebilir.
