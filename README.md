# MCSA Motor Fault Detection

Machine learning-based fault classification for three-phase induction motors using **Motor Current Signature Analysis (MCSA)**. The system detects **broken rotor bar (BRB)** and **stator winding fault (SWF)** conditions from motor current signals across multiple VFD-controlled operating speeds.

**Ali Ashoor** · General Assembly Data Science Bootcamp · DSB PT3 · Bahrain · 2026

---

## Overview

Motors are critical to industrial processes, and undetected faults can lead to unplanned downtime and costly failures. MCSA provides a non-invasive approach to motor condition monitoring by extracting diagnostic information from existing motor current measurements.

This project extends a previous senior thesis (Bahrain Polytechnic, 2025 — **Best Research Award**) that developed a rule-based MCSA fault detection system. The previous system achieved 100% detection at full speed but degraded substantially at reduced speeds.

This capstone investigates whether machine learning can provide more robust fault classification and examines how different feature representations affect model performance.

### Research Question

**Can machine learning classify motor faults from engineered MCSA features, which feature representation performs best, and how well does the resulting model generalize across operating speeds?**

---

## Results

| Model             | Pipeline          | Features | Weighted F1 |
| ----------------- | ----------------- | -------: | ----------: |
| **Random Forest** | **FFT (reduced)** |   **17** |  **0.8987** |
| Random Forest     | FFT (full)        |       43 |      0.8716 |
| SVM               | Time-Domain       |       10 |      0.8589 |
| XGBoost           | CWT               |        8 |      0.8490 |

**Recommended model:** Random Forest using 17 reduced FFT features.

**Per-class F1:** BRB **0.85** · Healthy **0.98** · SWF **0.86**

The Hybrid Random Forest achieved a higher conventional test score (**F1 = 0.9876**), but it was not selected because its performance collapsed under leave-one-speed-out testing, indicating poor generalization to unseen operating speeds.

---

## Comparison with the Rule-Based Baseline

The previous rule-based MCSA system achieved:

* **100%** at full speed
* **50%** at 75% speed
* **0%** at 50% speed

The recommended ML model achieved **weighted F1 = 0.8987** under the conventional three-speed test split.

The comparison suggests that the ML approach provides substantially more consistent classification across the evaluated operating conditions, although the ML and rule-based results should not be interpreted as identical metrics.

---

## Method

### Dataset

The dataset is based on Bruinsma et al. (2024), available through 4TU.ResearchData.

* 2,700,000 rows
* 18 columns
* One motor design (Motor 2)
* Conditions: Healthy, BRB, SWF
* Speeds: 50%, 75%, 100%
* Three current phases: A, B, C
* Five measurement replicates per condition/speed group
* Sampling frequency: 20 kHz
* Perfectly balanced classes

The data are stored in:

```text
data/mcsa_master_dataset.parquet
```

### Train/Test Split

The split was performed at the **measurement/recording level before windowing**.

Measurements m1–m4 were used for training and m5 for testing. This prevents overlapping windows from the same experimental recording from appearing in both training and test sets.

### Segmentation

Each recording was divided using:

* Window size: **1,024 samples**
* Overlap: **50%**
* Step size: **512 samples**

This produced:

* **63,072 training windows**
* **15,768 test windows**

### Feature Pipelines

| Pipeline | Features | Description                                                                             |
| -------- | -------: | --------------------------------------------------------------------------------------- |
| TD       |       10 | Time-domain statistics including RMS, kurtosis, crest factor, waveform length and ZCR   |
| FFT      |       43 | Frequency-domain features based on fault-related sidebands and harmonic characteristics |
| CWT      |        8 | Wavelet energy and entropy features from fault-related frequency zones                  |
| HYB      |       61 | Combination of TD, FFT and CWT features                                                 |

### Models

Four model families were evaluated across the four feature pipelines:

* Dummy Classifier
* Random Forest
* SVM
* XGBoost

This produced **16 pipeline/model combinations**.

The strongest candidates were then subjected to two-stage hyperparameter tuning using:

1. RandomizedSearchCV
2. GridSearchCV

Five-fold cross-validation was performed using the training data only.

### Evaluation

The primary metric was **weighted F1-score**, supported by:

* Per-class precision
* Per-class recall
* Confusion matrices
* Cross-speed generalization testing

---

## Key Findings

### 1. Raw amplitude shows negligible fault correlation

The maximum Pearson correlation between any raw current signal column and the fault label was only **0.007**.

This indicates negligible linear separation of the fault conditions using raw amplitude alone and motivated the use of frequency-domain and time-frequency features.

### 2. Frequency-domain features are important for this dataset

RMS strongly reflected operating speed but showed little separation between Healthy, BRB and SWF conditions at the same speed.

The fault signatures therefore required engineered frequency-domain information for effective classification.

### 3. Feature reduction outperformed hyperparameter tuning

For the FFT Random Forest:

* Hyperparameter tuning improved F1 by approximately **0.65 percentage points**
* Reducing the feature set from 43 to 17 improved F1 by approximately **3.36 percentage points**

This suggests that removing redundant or less useful features had a greater impact than further parameter optimization.

### 4. The highest-scoring HYB model did not generalize across speeds

The Hybrid Random Forest achieved:

**Conventional test F1 = 0.9876**

However, when evaluated using leave-one-speed-out testing, performance collapsed toward the three-class random-chance level of **0.3333**:

* HYB: **0.3660**
* FFT: **0.3844**

This indicates that strong conventional test performance does not necessarily imply robustness to unseen operating speeds.

### 5. Operating speed strongly affects frequency-based features

Under VFD control, supply frequency changes with operating speed:

**25 Hz → 37.5 Hz → 50 Hz**

Consequently, the spectral location of fault-related components also changes.

This creates a significant challenge for models trained on fixed operating speeds.

### 6. BRB detection is limited by FFT resolution

A 1,024-sample window at 20 kHz provides approximately **19.5 Hz frequency resolution**.

BRB sidebands can be only approximately **0.67–1.33 Hz** from the fundamental in the evaluated conditions.

Therefore, the windowed FFT cannot reliably resolve these closely spaced components.

The final 17-feature FFT model is consequently dominated by SWF-related features rather than BRB-specific sideband features.

---

## Final Model Selection

The **17-feature FFT Random Forest** was selected as the recommended model.

Selection was based on more than conventional test-set F1:

* Strong classification performance
* Better physical interpretability
* Reduced feature complexity
* Direct connection to established MCSA frequency-domain signatures
* More defensible engineering interpretation than the HYB model

The HYB model was not rejected because its classification performance was poor. It was rejected because its robustness to unseen operating speeds could not be established.

---

## Limitations

* **Single motor design** — the study uses one motor design, so results may not transfer directly to other motors.
* **Fixed operating speeds** — the models were evaluated on three operating speeds; unseen-speed generalization remains a limitation.
* **BRB resolution** — reliable detection of closely spaced BRB sidebands would require a longer measurement window or a higher-resolution frequency representation.
* **Limited fault coverage** — only Healthy, BRB and SWF conditions were studied.
* **Feature-selection optimism** — the 17-feature selection was informed by test-set performance. A stricter cross-validated selection produced F1 = **0.8929**.

---

## Repository Structure

```text
mcsa-motor-fault-detection/
├── data/
│   └── mcsa_master_dataset.parquet
├── img/
│   ├── sliding_window_overlap.png
│   └── why_overlap_matters.png
├── mcsa_capstone.ipynb
├── Motor-Fault-Detection-using-MCSA.pptx
├── Motor-Fault-Detection-using-MCSA.pdf
└── README.md
```

---

## Requirements

```text
Python 3.12
pandas 3.0.2
numpy 2.4.4
scipy 1.17.1
scikit-learn 1.8.0
xgboost 3.3.0
PyWavelets 1.8.0
joblib 1.5.3
matplotlib 3.10.8
seaborn 0.13.2
```

Run the notebook from top to bottom.

Full execution takes several hours, with the largest computational costs coming from CWT feature extraction and SVM hyperparameter tuning.

---

## References

Bruinsma, S., Geertsma, R. D., Loendersloot, R., & Tinga, T. (2023). *Motor current and vibration monitoring dataset for various faults in an E-motor-driven centrifugal pump.* Data in Brief, 52, 109987.

Ashoor, A., & Sanad, S. (2025). *Rule-Based MCSA Fault Detection System.* B.Sc. thesis, Bahrain Polytechnic. Supervised by Dr. Zakareya Hasan. Best Research Award 2025.

Full reference list is provided in Section 11 of the notebook.
