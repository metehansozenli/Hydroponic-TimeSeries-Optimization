// HydroGrow AI - Hidroponik Bitki Büyüme Simülatörü JavaScript
document.addEventListener('DOMContentLoaded', () => {
    // DOM Elemanları
    const form = document.getElementById('growth-form');
    const loadingEl = document.getElementById('loading');
    const initialMessageEl = document.getElementById('initial-message');
    const resultsContainerEl = document.getElementById('results-container');
    const environmentAssessmentEl = document.getElementById('environment-assessment');
    const growthChartEl = document.getElementById('growth-chart');
    
    // Debug: Element'lerin varlığını kontrol et
    console.log('Form element:', form);
    console.log('Loading element:', loadingEl);
    
    // Eğer form bulunamazsa hata ver
    if (!form) {
        console.error('Form element bulunamadı! ID: growth-form');
        return;
    }
    
    // Bitki Seçimi Elemanları
    const plantTypeInputs = document.querySelectorAll('input[name="plant_type"]');    const selectedPlantInfo = {
        basil: {
            name: 'Fesleğen',
            icon: '🌿',
            description: 'Aromatik ve hızlı büyüyen bir bitki. Hidroponik sistemlerde mükemmel sonuçlar verir.',
            characteristics: ['Hızlı büyüme', 'Yüksek verim', 'Aromalı yapraklar']
        },
        lettuce: {
            name: 'Marul',
            icon: '🥬',
            description: 'Yapraklı yeşil sebze. Hidroponik sistemlerde en popüler bitkilerden biri.',
            characteristics: ['Düşük bakım', 'Sürekli hasat', 'Besleyici']
        },
        strawberry: {
            name: 'Çilek',
            icon: '🍓',
            description: 'Tatlı meyveler veren bitki. Vertical farming için ideal.',
            characteristics: ['Tatlı meyveler', 'Uzun hasat süresi', 'Yüksek değer']
        },
        tomato: {
            name: 'Domates',
            icon: '🍅',
            description: 'Büyük ve verimli bitkiler. Profesyonel hidroponik üretim için mükemmel.',
            characteristics: ['Büyük bitkiler', 'Yüksek verim', 'Uzun dönem']
        }
    };
      // Parametre Kontrolleri
    const tempInput = document.getElementById('h2o_temp');
    const phInput = document.getElementById('ph');
    const ecInput = document.getElementById('ec');
    const daysInput = document.getElementById('days');
    
    // Debug: Slider element'lerinin varlığını kontrol et
    console.log('Temp slider:', tempInput);
    console.log('pH slider:', phInput);
    console.log('EC slider:', ecInput);
    console.log('Days slider:', daysInput);
    
    // Değer Göstergeleri
    const tempValueEl = document.getElementById('h2o_temp_value');
    const phValueEl = document.getElementById('ph_value');
    const ecValueEl = document.getElementById('ec_value');
    const daysValueEl = document.getElementById('days_value');
    
    // Debug: Value element'lerinin varlığını kontrol et
    console.log('Temp value element:', tempValueEl);
    console.log('pH value element:', phValueEl);
    console.log('EC value element:', ecValueEl);
    console.log('Days value element:', daysValueEl);
      // Sonuç Elemanları
    const assessmentTextEl = document.getElementById('assessment-text');
    const assessmentAlertEl = document.getElementById('assessment-alert');
    const scorePercentageEl = document.getElementById('score-percentage');
    const startHeightEl = document.getElementById('start-height');
    const finalHeightEl = document.getElementById('final-height');
    const growthRateEl = document.getElementById('growth-rate');
    const performanceScoreEl = document.getElementById('performance-score');
    
    // Debug: Metric element'lerinin varlığını kontrol et
    console.log('Start height element:', startHeightEl);
    console.log('Final height element:', finalHeightEl);
    console.log('Growth rate element:', growthRateEl);
    
    // Canlı Göstergeler
    const liveTempEl = document.getElementById('live-temp');
    const livePhEl = document.getElementById('live-ph');
    const liveEcEl = document.getElementById('live-ec');
    const heightIndicatorEl = document.getElementById('current-height-indicator');
      // Slider Değişiklikleri - Gerçek Zamanlı Güncelleme
    if (tempInput && tempValueEl) {
        tempInput.addEventListener('input', () => {
            const value = parseFloat(tempInput.value).toFixed(1);
            tempValueEl.textContent = `${value}°C`;
            if (liveTempEl) liveTempEl.textContent = `${value}°C`;
            updateEnvironmentAssessmentRealTime();
            updateSliderStyle(tempInput);
        });
    } else {
        console.error('Temp slider veya value elementi bulunamadı!');
    }
    
    phInput.addEventListener('input', () => {
        const value = parseFloat(phInput.value).toFixed(2);
        phValueEl.textContent = value;
        if (livePhEl) livePhEl.textContent = value;
        updateEnvironmentAssessmentRealTime();
        updateSliderStyle(phInput);
    });
      ecInput.addEventListener('input', () => {
        const value = parseFloat(ecInput.value).toFixed(2);
        ecValueEl.textContent = `${value} mS/cm`;
        if (liveEcEl) liveEcEl.textContent = `${value} mS/cm`;
        updateEnvironmentAssessmentRealTime();
        updateSliderStyle(ecInput);
    });
    
    daysInput.addEventListener('input', () => {
        daysValueEl.textContent = `${daysInput.value} gün`;
    });
    
    // Bitki Türü Seçimi Event Listener
    plantTypeInputs.forEach(input => {
        input.addEventListener('change', (e) => {
            const selectedPlant = e.target.value;
            updatePlantInfo(selectedPlant);
        });
    });
    
    // Bitki Bilgilerini Güncelle
    function updatePlantInfo(plantType) {
        const plantInfo = selectedPlantInfo[plantType];
        if (plantInfo) {
            // Bitki seçimi başarılı - UI güncelleme burada yapılabilir
            console.log(`Seçilen bitki: ${plantInfo.name}`);
        }
    }
    
    // Slider Stili Güncelle (Renk Gradyanı)
    function updateSliderStyle(slider) {
        const min = parseFloat(slider.min);
        const max = parseFloat(slider.max);
        const value = parseFloat(slider.value);
        const percentage = ((value - min) / (max - min)) * 100;
        
        slider.style.background = `linear-gradient(to right, #4CAF50 0%, #4CAF50 ${percentage}%, #e3f2fd ${percentage}%, #e3f2fd 100%)`;
    }    // Form Gönderimi - Simülasyon Başlatma
    if (form) {
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            // Yükleme Göster
            showLoading();
              const formData = new FormData(form);
            
            try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (data.error) {
                showError(data.error);
                return;
            }
            
            displayResults(data);
              } catch (error) {
            console.error('Simülasyon Hatası:', error);
            showError('Simülasyon sırasında bir hata oluştu. Lütfen parametreleri kontrol edip tekrar deneyin.');
        } finally {
            hideLoading();
        }
    });
    } else {
        console.error('Form elementi bulunamadı! Simülasyon başlatılamaz.');
    }
    
    // Yükleme Göster
    function showLoading() {
        loadingEl.classList.remove('d-none');
        initialMessageEl.classList.add('d-none');
        resultsContainerEl.classList.add('d-none');
        
        // Yükleme animasyonu başlat
        startLoadingAnimation();
    }
    
    // Yükleme Gizle
    function hideLoading() {
        loadingEl.classList.add('d-none');
    }
    
    // Hata Göster
    function showError(message) {
        loadingEl.classList.add('d-none');
        initialMessageEl.innerHTML = `
            <div class="text-center py-5">
                <i class="fas fa-exclamation-triangle fa-4x text-warning mb-3"></i>
                <h4 class="text-danger">Simülasyon Hatası</h4>
                <p class="text-muted">${message}</p>
                <button class="btn btn-hydro mt-3" onclick="location.reload()">
                    <i class="fas fa-redo me-2"></i>Tekrar Dene
                </button>
            </div>
        `;
        initialMessageEl.classList.remove('d-none');
    }
    
    // Yükleme Animasyonu
    function startLoadingAnimation() {
        const steps = [
            'Hidroponik sistem analiz ediliyor...',
            'LSTM AI modeli çalıştırılıyor...',
            'Beslenme çözeltisi değerlendiriliyor...',
            'Büyüme tahmini hesaplanıyor...',
            'Grafik oluşturuluyor...'
        ];
        
        let currentStep = 0;
        const loadingText = loadingEl.querySelector('p');
        
        const interval = setInterval(() => {
            if (currentStep < steps.length) {
                loadingText.textContent = steps[currentStep];
                currentStep++;
            } else {
                loadingText.textContent = 'Simülasyon tamamlanıyor...';
                clearInterval(interval);
            }
        }, 800);
    }
      // Sonuçları Göster
    function displayResults(data) {
        // Hata kontrolü
        if (data.error) {
            showError(data.error);
            return;
        }
        
        console.log('Results container bulundu:', resultsContainerEl);
        console.log('Results container görünür yapılıyor...');
        
        resultsContainerEl.classList.remove('d-none');
        resultsContainerEl.style.display = 'block';
        resultsContainerEl.style.visibility = 'visible';
        resultsContainerEl.style.opacity = '1';
        
        console.log('Results container class list:', resultsContainerEl.classList.toString());
        console.log('Results container display:', window.getComputedStyle(resultsContainerEl).display);
        
        // Sistem Analizi Güncelle
        updateSystemAssessment(data);
        
        // Grafik Göster
        if (data.chart_img) {
            growthChartEl.src = `data:image/png;base64,${data.chart_img}`;
        }
        
        // İstatistikleri Güncelle
        updateGrowthStatistics(data);
        
        // Bitki Bilgilerini Göster
        updatePlantDisplay(data.plant_info);
        
        // Bitki Animasyonu Başlat
        startPlantGrowthAnimation(data);
        
        // Canlı Parametreleri Güncelle
        updateLiveParameters();
        
        // Sonuçları kaydet (analitik için)
        saveSimulationResults(data);
    }
    
    // Sistem Analizi Güncelle
    function updateSystemAssessment(data) {
        const score = data.comparison?.overall_score || 0;
        
        assessmentTextEl.textContent = data.assessment || 'Hidroponik sistem analizi tamamlandı';
        
        if (scorePercentageEl) {
            scorePercentageEl.textContent = `${Math.round(score)}%`;
        }
        
        // Performans durumuna göre stil ayarla
        const alertClass = getPerformanceAlertClass(score);
        assessmentAlertEl.className = `performance-summary ${alertClass}`;
        
        // Performans ikonu güncelle
        const icon = assessmentAlertEl.querySelector('.performance-icon i');
        if (icon) {
            icon.className = `fas ${getPerformanceIcon(score)} fa-2x`;
        }
    }
    
    // Büyüme İstatistikleri Güncelle
    function updateGrowthStatistics(data) {
        if (data.heights && data.heights.length > 0) {
            const startHeight = data.heights[0];
            const finalHeight = data.heights[data.heights.length - 1];
            const growth = finalHeight - startHeight;
            const days = data.heights.length;
            const avgGrowthRate = growth / days;
              startHeightEl.textContent = `${startHeight.toFixed(1)} cm`;
            // Zorla görünür yap
            startHeightEl.style.display = 'block';
            startHeightEl.style.visibility = 'visible';
            startHeightEl.style.opacity = '1';
            startHeightEl.style.color = '#1976D2';
            startHeightEl.style.fontSize = '2em';
            startHeightEl.style.fontWeight = 'bold';
            
            // Debug: finalHeightEl kontrolü
            console.log('finalHeightEl:', finalHeightEl);
            console.log('finalHeight value:', finalHeight);            if (finalHeightEl) {
                finalHeightEl.textContent = `${finalHeight.toFixed(1)} cm`;
                // Zorla görünür yap
                finalHeightEl.style.display = 'block';
                finalHeightEl.style.visibility = 'visible';
                finalHeightEl.style.opacity = '1';
                finalHeightEl.style.color = '#1976D2';
                finalHeightEl.style.fontSize = '2em';
                finalHeightEl.style.fontWeight = 'bold';
                console.log('Final height updated:', finalHeightEl.textContent);
            } else {
                console.error('finalHeightEl bulunamadı!');
            }
              if (growthRateEl) {
                growthRateEl.textContent = `${avgGrowthRate.toFixed(2)} cm/gün`;
                // Zorla görünür yap
                growthRateEl.style.display = 'block';
                growthRateEl.style.visibility = 'visible';
                growthRateEl.style.opacity = '1';
                growthRateEl.style.color = '#4CAF50';
                growthRateEl.style.fontSize = '2em';
                growthRateEl.style.fontWeight = 'bold';
                console.log('Growth rate updated:', growthRateEl.textContent);
                console.log('Growth rate element visibility:', window.getComputedStyle(growthRateEl).visibility);
            } else {
                console.error('growthRateEl bulunamadı!');
            }
              if (data.comparison?.overall_score) {
                if (performanceScoreEl) {
                    performanceScoreEl.textContent = `${Math.round(data.comparison.overall_score)}%`;
                    // Zorla görünür yap
                    performanceScoreEl.style.display = 'block';
                    performanceScoreEl.style.visibility = 'visible';
                    performanceScoreEl.style.opacity = '1';
                    performanceScoreEl.style.color = '#FF9800';
                    performanceScoreEl.style.fontSize = '2em';
                    performanceScoreEl.style.fontWeight = 'bold';
                    console.log('Performance score updated:', performanceScoreEl.textContent);
                    console.log('Performance score element visibility:', window.getComputedStyle(performanceScoreEl).visibility);
                } else {
                    console.error('performanceScoreEl bulunamadı!');
                }
            } else {
                console.log('Overall score verisi yok:', data.comparison);
            }
              // Metrik kartlarını animasyonla güncelle
            animateMetricCards();
        }
    }
    
    // Bitki Bilgilerini Göster
    function updatePlantDisplay(plantInfo) {
        if (!plantInfo) return;
        
        // Bitki türü bilgisini konsola yazdır (debug için)
        console.log(`Simülasyon tamamlandı: ${plantInfo.name} (${plantInfo.type})`);
        
        // Gelecekte buraya bitki türüne özel UI güncellemeleri eklenebilir
        // Örneğin: farklı bitki türleri için farklı animasyon stilleri
    }
    
    // Bitki Büyüme Animasyonu
    function startPlantGrowthAnimation(data) {
        if (!data.heights || data.heights.length === 0) return;
        
        const container = document.getElementById('plant-animation');
        if (!container) return;
        
        // Eski animasyonu temizle
        container.innerHTML = '';
        
        const finalHeight = data.heights[data.heights.length - 1];
        const maxContainerHeight = 250; // piksel cinsinden
        const scale = Math.min(maxContainerHeight / finalHeight, 4); // Maksimum ölçek
        
        // Bitki elementi oluştur
        createHydroPlant(container, finalHeight, scale);
        
        // Büyüme animasyonu başlat
        animateGrowth(container, data.heights, scale);
        
        // Cetveldeki yükseklik göstergesi
        updateHeightIndicator(finalHeight);
    }
    
    // Hidroponik Bitki Oluştur
    function createHydroPlant(container, finalHeight, scale) {
        // Kök sistemi
        const roots = document.createElement('div');
        roots.className = 'plant-roots';
        roots.style.cssText = `
            position: absolute;
            bottom: -20px;
            left: 50%;
            transform: translateX(-50%);
            width: 60px;
            height: 40px;
            background: radial-gradient(ellipse, rgba(139, 195, 74, 0.6), transparent);
            border-radius: 50%;
            opacity: 0;
        `;
        container.appendChild(roots);
        
        // Ana gövde
        const stem = document.createElement('div');
        stem.className = 'plant-stem';
        stem.style.cssText = `
            position: absolute;
            bottom: 0;
            left: 50%;
            transform: translateX(-50%);
            width: 8px;
            height: 0;
            background: linear-gradient(to top, #6B8E23, #8BC34A);
            border-radius: 4px;
            box-shadow: 0 0 10px rgba(107, 142, 35, 0.3);
        `;
        container.appendChild(stem);
        
        // Yapraklar için konteyner
        const leavesContainer = document.createElement('div');
        leavesContainer.className = 'leaves-container';
        leavesContainer.style.position = 'relative';
        container.appendChild(leavesContainer);
        
        return { roots, stem, leavesContainer };
    }
    
    // Büyüme Animasyonu
    function animateGrowth(container, heights, scale) {
        const stem = container.querySelector('.plant-stem');
        const roots = container.querySelector('.plant-roots');
        const leavesContainer = container.querySelector('.leaves-container');
        
        if (!stem) return;
        
        const finalHeight = heights[heights.length - 1] * scale;
        const duration = 4; // saniye
        
        // Kök animasyonu
        if (roots) {
            gsap.to(roots, {
                opacity: 1,
                scale: 1.2,
                duration: duration * 0.3,
                ease: "power2.out"
            });
        }
        
        // Gövde büyüme animasyonu
        gsap.to(stem, {
            height: finalHeight,
            duration: duration,
            ease: "power1.out",
            onUpdate: function() {
                // Büyüme sırasında yaprak ekle
                const currentHeight = parseFloat(stem.style.height);
                if (currentHeight > 0 && leavesContainer) {
                    updateLeaves(leavesContainer, currentHeight, finalHeight);
                }
            }
        });
        
        // Çiçek/meyve animasyonu (büyüme tamamlandığında)
        setTimeout(() => {
            addFlowerOrFruit(container, finalHeight);
        }, duration * 1000);
    }
    
    // Yaprak Güncelleme
    function updateLeaves(container, currentHeight, finalHeight) {
        const leafLevels = [0.3, 0.5, 0.7, 0.9]; // Yükseklik yüzdeleri
        
        leafLevels.forEach((level, index) => {
            const targetHeight = finalHeight * level;
            
            if (currentHeight >= targetHeight && !container.querySelector(`[data-leaf-level="${index}"]`)) {
                createLeafPair(container, targetHeight, index);
            }
        });
    }
    
    // Yaprak Çifti Oluştur
    function createLeafPair(container, height, level) {
        const leafPair = document.createElement('div');
        leafPair.setAttribute('data-leaf-level', level);
        leafPair.style.position = 'absolute';
        leafPair.style.bottom = `${height}px`;
        leafPair.style.left = '50%';
        leafPair.style.transform = 'translateX(-50%)';
        
        // Sol yaprak
        const leftLeaf = document.createElement('div');
        leftLeaf.style.cssText = `
            position: absolute;
            left: -25px;
            width: 20px;
            height: 30px;
            background: linear-gradient(45deg, #4CAF50, #8BC34A);
            border-radius: 50% 10% 50% 10%;
            transform: rotate(-45deg);
            box-shadow: 0 2px 8px rgba(76, 175, 80, 0.3);
            opacity: 0;
        `;
        
        // Sağ yaprak
        const rightLeaf = document.createElement('div');
        rightLeaf.style.cssText = `
            position: absolute;
            right: -25px;
            width: 20px;
            height: 30px;
            background: linear-gradient(45deg, #4CAF50, #8BC34A);
            border-radius: 10% 50% 10% 50%;
            transform: rotate(45deg);
            box-shadow: 0 2px 8px rgba(76, 175, 80, 0.3);
            opacity: 0;
        `;
        
        leafPair.appendChild(leftLeaf);
        leafPair.appendChild(rightLeaf);
        container.appendChild(leafPair);
        
        // Yaprak animasyonu
        gsap.to([leftLeaf, rightLeaf], {
            opacity: 1,
            scale: 1,
            duration: 0.8,
            ease: "back.out(1.7)",
            stagger: 0.1
        });
        
        // Hafif sallanma animasyonu
        gsap.to(leftLeaf, {
            rotation: -40,
            duration: 2,
            repeat: -1,
            yoyo: true,
            ease: "sine.inOut"
        });
        
        gsap.to(rightLeaf, {
            rotation: 40,
            duration: 2.2,
            repeat: -1,
            yoyo: true,
            ease: "sine.inOut"
        });
    }
    
    // Çiçek/Meyve Ekle
    function addFlowerOrFruit(container, height) {
        const flower = document.createElement('div');
        flower.style.cssText = `
            position: absolute;
            bottom: ${height + 10}px;
            left: 50%;
            transform: translateX(-50%);
            width: 16px;
            height: 16px;
            background: radial-gradient(circle, #FF5722, #E64A19);
            border-radius: 50%;
            box-shadow: 0 0 15px rgba(255, 87, 34, 0.6);
            opacity: 0;
        `;
        
        container.appendChild(flower);
        
        gsap.to(flower, {
            opacity: 1,
            scale: 1.5,
            duration: 1,
            ease: "bounce.out"
        });
    }    
    // Yükseklik Göstergesi Güncelle
    function updateHeightIndicator(height) {
        if (heightIndicatorEl) {
            heightIndicatorEl.textContent = `${height.toFixed(1)} cm`;
            
            // Animasyonla yükseklik göstergesi pozisyonunu güncelle
            gsap.to(heightIndicatorEl, {
                bottom: `${Math.min(height * 3, 280)}px`, // Ölçekli pozisyon
                duration: 2,
                ease: "power1.out"
            });
        }
    }
    
    // Canlı Parametreleri Güncelle
    function updateLiveParameters() {
        if (liveTempEl) liveTempEl.textContent = `${parseFloat(tempInput.value).toFixed(1)}°C`;
        if (livePhEl) livePhEl.textContent = parseFloat(phInput.value).toFixed(2);
        if (liveEcEl) liveEcEl.textContent = `${parseFloat(ecInput.value).toFixed(2)} mS/cm`;
    }
    
    // Metrik Kartları Animasyonu
    function animateMetricCards() {
        const metricCards = document.querySelectorAll('.metric-card');
        
        // GSAP kontrolü
        if (typeof gsap !== 'undefined') {
            gsap.from(metricCards, {
                y: 30,
                opacity: 0,
                duration: 0.8,
                stagger: 0.2,
                ease: "back.out(1.7)"
            });
        } else {
            // GSAP yoksa basit animasyon
            console.log('GSAP bulunamadı, basit animasyon kullanılıyor');
            metricCards.forEach((card, index) => {
                card.style.opacity = '1';
                card.style.visibility = 'visible';
                card.style.display = 'block';
                card.style.transform = 'translateY(0)';
            });
        }
    }
    
    // Performans Durumu Sınıfı
    function getPerformanceAlertClass(score) {
        if (score >= 90) return 'alert-success';
        if (score >= 70) return 'alert-info';
        if (score >= 50) return 'alert-warning';
        return 'alert-danger';
    }
    
    // Performans İkonu
    function getPerformanceIcon(score) {
        if (score >= 90) return 'fa-check-circle';
        if (score >= 70) return 'fa-thumbs-up';
        if (score >= 50) return 'fa-exclamation-circle';
        return 'fa-exclamation-triangle';
    }
    
    // Gerçek Zamanlı Ortam Değerlendirmesi
    function updateEnvironmentAssessmentRealTime() {
        const h2o_temp = parseFloat(tempInput.value);
        const ph = parseFloat(phInput.value);
        const ec = parseFloat(ecInput.value);
        
        // Basit değerlendirme (optimum değerler için)
        const optimalTemp = 22.0; // Örnek optimal değer
        const optimalPh = 6.2;    // Örnek optimal değer
        const optimalEc = 2.5;    // Örnek optimal değer
        
        const tempScore = calculateParameterScore(h2o_temp, optimalTemp, 19, 27);
        const phScore = calculateParameterScore(ph, optimalPh, 5.4, 7.4);
        const ecScore = calculateParameterScore(ec, optimalEc, 1.6, 3.8);
        
        const overallScore = (tempScore + phScore + ecScore) / 3;
        
        // Hidroponik durum HTML'i
        const statusHtml = `
            <div class="hydro-status-display">
                <div class="status-header mb-3">
                    <h5 class="text-center">
                        <i class="fas fa-flask me-2 text-info"></i>
                        Beslenme Çözeltisi Durumu
                    </h5>
                </div>
                
                <div class="row">
                    <div class="col-4">
                        <div class="param-status ${getStatusClass(tempScore)}">
                            <div class="param-icon">
                                <i class="fas fa-thermometer-half"></i>
                            </div>
                            <div class="param-value">${h2o_temp.toFixed(1)}°C</div>
                            <div class="param-label">Sıcaklık</div>
                            <div class="param-bar">
                                <div class="param-fill" style="width: ${tempScore}%"></div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="col-4">
                        <div class="param-status ${getStatusClass(phScore)}">
                            <div class="param-icon">
                                <i class="fas fa-vial"></i>
                            </div>
                            <div class="param-value">${ph.toFixed(2)}</div>
                            <div class="param-label">pH</div>
                            <div class="param-bar">
                                <div class="param-fill" style="width: ${phScore}%"></div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="col-4">
                        <div class="param-status ${getStatusClass(ecScore)}">
                            <div class="param-icon">
                                <i class="fas fa-bolt"></i>
                            </div>
                            <div class="param-value">${ec.toFixed(2)}</div>
                            <div class="param-label">EC (mS/cm)</div>
                            <div class="param-bar">
                                <div class="param-fill" style="width: ${ecScore}%"></div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="overall-status mt-4 text-center">
                    <div class="status-circle ${getOverallStatusClass(overallScore)}">
                        <div class="status-percentage">${Math.round(overallScore)}%</div>
                        <div class="status-text">Sistem Verimliliği</div>
                    </div>
                    
                    <div class="status-advice mt-3">
                        <i class="fas ${getAdviceIcon(overallScore)} me-2"></i>
                        <span class="advice-text">${getAdviceText(overallScore)}</span>
                    </div>
                </div>
            </div>
        `;
        
        if (environmentAssessmentEl) {
            environmentAssessmentEl.innerHTML = statusHtml;
        }
    }
    
    // Parametre Skoru Hesapla
    function calculateParameterScore(value, optimal, min, max) {
        const range = max - min;
        const distance = Math.abs(value - optimal);
        const maxDistance = Math.max(optimal - min, max - optimal);
        const score = Math.max(0, 100 - (distance / maxDistance) * 100);
        return Math.round(score);
    }
    
    // Durum Sınıfı
    function getStatusClass(score) {
        if (score >= 90) return 'status-excellent';
        if (score >= 70) return 'status-good';
        if (score >= 50) return 'status-average';
        return 'status-poor';
    }
    
    // Genel Durum Sınıfı
    function getOverallStatusClass(score) {
        if (score >= 90) return 'overall-excellent';
        if (score >= 70) return 'overall-good';
        if (score >= 50) return 'overall-average';
        return 'overall-poor';
    }
    
    // Tavsiye İkonu
    function getAdviceIcon(score) {
        if (score >= 90) return 'fa-check-circle';
        if (score >= 70) return 'fa-thumbs-up';
        if (score >= 50) return 'fa-exclamation-circle';
        return 'fa-exclamation-triangle';
    }
    
    // Tavsiye Metni
    function getAdviceText(score) {
        if (score >= 90) return 'Mükemmel! Hidroponik sistem optimal durumda.';
        if (score >= 70) return 'İyi durumda. Küçük ayarlamalar yapabilirsiniz.';
        if (score >= 50) return 'Orta seviye. Parametreleri optimize etmeyi deneyin.';
        return 'Dikkat! Parametreler optimal aralığın dışında.';
    }
    
    // Simülasyon Sonuçlarını Kaydet (Analytics)
    function saveSimulationResults(data) {
        const simulationData = {
            timestamp: new Date().toISOString(),
            parameters: {
                temperature: parseFloat(tempInput.value),
                ph: parseFloat(phInput.value),
                ec: parseFloat(ecInput.value),
                days: parseInt(daysInput.value)
            },
            results: {
                finalHeight: data.heights ? data.heights[data.heights.length - 1] : 0,
                growthRate: data.growth_rate || 0,
                score: data.comparison?.overall_score || 0
            }
        };
        
        // localStorage'a kaydet (gelecekteki analitik için)
        const existingData = JSON.parse(localStorage.getItem('hydroGrowSimulations') || '[]');
        existingData.push(simulationData);
        
        // Son 50 simülasyonu sakla
        if (existingData.length > 50) {
            existingData.splice(0, existingData.length - 50);
        }
        
        localStorage.setItem('hydroGrowSimulations', JSON.stringify(existingData));
    }
    
    // Başlangıçta slider stillerini güncelle
    function initializeSliders() {
        updateSliderStyle(tempInput);
        updateSliderStyle(phInput);
        updateSliderStyle(ecInput);
        updateSliderStyle(daysInput);
        
        // İlk değerleri güncelle
        tempValueEl.textContent = `${parseFloat(tempInput.value).toFixed(1)}°C`;
        phValueEl.textContent = parseFloat(phInput.value).toFixed(2);
        ecValueEl.textContent = `${parseFloat(ecInput.value).toFixed(2)} mS/cm`;
        daysValueEl.textContent = `${daysInput.value} gün`;
    }
    
    // Sayfa yüklendiğinde başlangıç ayarları
    initializeSliders();
    updateEnvironmentAssessmentRealTime();
    updateLiveParameters();
    
    // Konsol mesajı
    console.log('🌱 HydroGrow AI Simülatörü başarıyla yüklendi!');
    console.log('💡 Hidroponik tarımda yapay zeka destekli büyüme optimizasyonu');
});

// CSS Stilleri - Dinamik Olarak Eklenen Stiller
const dynamicStyles = `
<style>
.hydro-status-display {
    padding: 20px;
}

.param-status {
    text-align: center;
    padding: 15px;
    border-radius: 15px;
    margin-bottom: 10px;
    transition: all 0.3s ease;
}

.param-status.status-excellent {
    background: linear-gradient(135deg, rgba(76, 175, 80, 0.1), rgba(76, 175, 80, 0.2));
    border: 2px solid rgba(76, 175, 80, 0.3);
}

.param-status.status-good {
    background: linear-gradient(135deg, rgba(33, 150, 243, 0.1), rgba(33, 150, 243, 0.2));
    border: 2px solid rgba(33, 150, 243, 0.3);
}

.param-status.status-average {
    background: linear-gradient(135deg, rgba(255, 193, 7, 0.1), rgba(255, 193, 7, 0.2));
    border: 2px solid rgba(255, 193, 7, 0.3);
}

.param-status.status-poor {
    background: linear-gradient(135deg, rgba(244, 67, 54, 0.1), rgba(244, 67, 54, 0.2));
    border: 2px solid rgba(244, 67, 54, 0.3);
}

.param-icon {
    font-size: 1.5em;
    margin-bottom: 8px;
    color: #2E7D32;
}

.param-value {
    font-size: 1.2em;
    font-weight: bold;
    color: #1976D2;
    margin-bottom: 5px;
}

.param-label {
    font-size: 0.85em;
    color: #666;
    margin-bottom: 8px;
}

.param-bar {
    width: 100%;
    height: 4px;
    background: rgba(0,0,0,0.1);
    border-radius: 2px;
    overflow: hidden;
}

.param-fill {
    height: 100%;
    background: linear-gradient(90deg, #4CAF50, #8BC34A);
    border-radius: 2px;
    transition: width 0.5s ease;
}

.status-circle {
    display: inline-block;
    width: 80px;
    height: 80px;
    border-radius: 50%;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    margin: 0 auto;
    color: white;
    font-weight: bold;
}

.overall-excellent {
    background: linear-gradient(135deg, #4CAF50, #45a049);
    box-shadow: 0 0 20px rgba(76, 175, 80, 0.4);
}

.overall-good {
    background: linear-gradient(135deg, #2196F3, #1976D2);
    box-shadow: 0 0 20px rgba(33, 150, 243, 0.4);
}

.overall-average {
    background: linear-gradient(135deg, #FF9800, #F57C00);
    box-shadow: 0 0 20px rgba(255, 152, 0, 0.4);
}

.overall-poor {
    background: linear-gradient(135deg, #F44336, #D32F2F);
    box-shadow: 0 0 20px rgba(244, 67, 54, 0.4);
}

.status-percentage {
    font-size: 1.2em;
    line-height: 1;
}

.status-text {
    font-size: 0.6em;
    line-height: 1;
    margin-top: 2px;
}

.status-advice {
    padding: 10px 15px;
    background: rgba(255, 255, 255, 0.9);
    border-radius: 20px;
    color: #2E7D32;
    font-weight: 600;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.advice-text {
    font-size: 0.9em;
}
</style>
`;

// Dinamik stilleri head'e ekle
document.head.insertAdjacentHTML('beforeend', dynamicStyles);
