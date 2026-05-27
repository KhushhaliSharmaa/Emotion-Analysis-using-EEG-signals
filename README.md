# 🧠 Emotion Analysis using EEG Signals

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Dataset](https://img.shields.io/badge/Dataset-DEAP-FF6B6B?style=for-the-badge)](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/)
[![Thesis](https://img.shields.io/badge/Bachelor's_Thesis-IIT_BHU-FF6B35?style=for-the-badge)](https://www.iitbhu.ac.in/)
[![Grade](https://img.shields.io/badge/Grade-10.0%2F10.0-gold?style=for-the-badge&logo=star&logoColor=white)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](https://opensource.org/licenses/MIT)

*A research-oriented implementation for analyzing and classifying emotional states from EEG brain signals using machine learning.*

</div>

---

## 📌 Table of Contents

1. [Overview](#-overview)
2. [Motivation](#-motivation)
3. [Dataset](#-dataset)
4. [Project Workflow](#-project-workflow)
5. [Repository Structure](#-repository-structure)
6. [Results & Evaluation](#-results--evaluation)
7. [Usage](#️-usage)
8. [Dependencies](#-dependencies)
9. [Related Publication](#-related-publication)
10. [License](#-license)

---

## 🚀 Overview

This repository contains the implementation for an end-to-end **EEG-based emotion classification system** developed as part of a Bachelor's Thesis at **IIT BHU**, awarded the highest possible grade of **10.0/10.0**.

The system:
- Processes EEG data collected during emotion-eliciting stimuli
- Extracts time-domain and frequency-domain features
- Applies PCA for dimensionality reduction
- Trains SVM classifiers to distinguish emotional states in the valence-arousal space

---

## 📊 Motivation

Understanding the relationship between neural activity and emotional states can:

- Advance research in **affective computing** and brain-computer interfaces
- Enable **real-time emotion detection** in HCI and therapeutic applications
- Provide insights into neural correlates of emotions for psychology and neuroscience

---

## 📁 Dataset

This project uses the **[DEAP Dataset](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/)** — a widely-used benchmark for EEG-based emotion research.

| Property | Details |
|---|---|
| **Subjects** | 32 participants |
| **Stimuli** | 40 one-minute music video clips |
| **Channels** | 32 EEG channels |
| **Labels** | Valence · Arousal · Dominance · Liking · Familiarity (1–9 scale) |
| **Emotion Classes** | HAHV · HALV · LAHV · LALV |

> ⚠️ DEAP is not included in this repo — download it separately from the [DEAP website](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/) and place it in the appropriate data directory.

---

## 🧠 Project Workflow

```
DEAP Dataset
     ↓
Data Preprocessing
(Bandpass filtering · Baseline removal · Artifact rejection)
     ↓
Feature Extraction
(Time-domain · Frequency-domain band powers · 1-second windows)
     ↓
Dimensionality Reduction
(PCA — 95% variance retained)
     ↓
SVM Classification
(One-vs-All strategy · Valence-Arousal model · 4 emotion classes)
     ↓
Evaluation
(Classification accuracy · 32 subjects)
```

---

## 📂 Repository Structure

```
Emotion-Analysis-using-EEG-signals/
│
├── FA(one vs all).py                        # Beta-band SVM classifier (one-vs-all)
├── FeatureALL(one vs all).py                # All-features SVM classifier
├── PCA_windows.py                           # PCA + feature windowing
├── Feature Extraction & Selection.ipynb     # Feature extraction notebook
├── Filtered EEG Signal_S01_Trail-10.ipynb  # EEG signal filtering notebook
└── README.md
```

---

## 📈 Results & Evaluation

Model performance evaluated on **32 subjects** using a one-vs-all classification strategy based on the **valence-arousal model**.

### Average Classification Accuracy (%)

| Features | HAHV | LAHV | HALV | LALV |
|---|---|---|---|---|
| FA (Beta Band only) | 50.00 | 52.50 | 48.44 | 47.81 |
| **All Features** | **58.70** | **79.35** | **64.83** | **72.58** |

**Key insights:**
- Using **all features** consistently outperforms beta-band-only features
- Best performance observed for the **LAHV** emotional state (79.35%)
- Beta-band features alone yield near-chance accuracy — multi-feature representations are essential

> Accuracies are averaged across 32 subjects.

### ⚠️ Limitations
- Performance may vary across subjects due to EEG inter-subject variability
- Model evaluated on DEAP only — no cross-dataset validation

### 🔮 Future Work
- Explore deep learning models (CNNs, LSTMs) for emotion classification
- Cross-subject and cross-dataset evaluation
- Real-time EEG emotion recognition

---

## ⚙️ Usage

**1. Clone the repository**
```bash
git clone https://github.com/KhushhaliSharmaa/Emotion-Analysis-using-EEG-signals.git
cd Emotion-Analysis-using-EEG-signals
```

**2. Prepare the dataset**
- Download the DEAP dataset from [here](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/)
- Place data in a folder (e.g., `data/DEAP/`)

**3. Run preprocessing & feature extraction**
```bash
python PCA_windows.py
```

**4. Train & evaluate models**
```bash
python "FA(one vs all).py"        # Beta-band features
python "FeatureALL(one vs all).py" # All features
```

---

## 🛠️ Dependencies

```bash
pip install numpy scipy scikit-learn pandas matplotlib mne
```

![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=flat-square&logo=scipy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square&logo=python&logoColor=white)

---

## 📄 Related Publication

This thesis was extended into a peer-reviewed publication:

**Sharma, K., Dash, A., Kumar, D.** — *"Investigating the Effect of EEG Channel Selection on Inter-subject Emotion Classification"*

| | |
|---|---|
| **Venue** | IEEE Confluence 2023 |
| **DOI** | [10.1109/Confluence56041.2023.10048851](https://doi.org/10.1109/Confluence56041.2023.10048851) |
| **Repository** | [Effect-of-Sensor-Selection-and-Features-on-Inter-Subject-emotion-classification](https://github.com/KhushhaliSharmaa/Effect-of-Sensor-Selection-and-Features-on-Inter-Subject-emotion-classification) |

---

## 📄 License

This project is licensed under the MIT License.
