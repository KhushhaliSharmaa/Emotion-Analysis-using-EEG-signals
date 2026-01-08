# ⭐ Emotion Analysis using EEG Signals
A research-oriented implementation for analyzing and classifying emotional states from EEG data using machine learning approaches. This project explores how neural signals correlate with emotional states by leveraging features extracted from EEG recordings and supervised classifiers.


# 📌 Table of Contents
1. 🚀 Overview
2. 📊 Motivation
3. 📁 Dataset
4. 🧠 Project Workflow
5. 📂 Repository Structure
6. ⚙️ Usage
7. 🛠 Dependencies
8. 📈 Results & Evaluation
9. 🤝 Contributing
10. 📜 License

# 🚀 Overview
This repository contains the code and documentation for an EEG-based emotion analysis system that:
Analyzes EEG data collected during emotion-eliciting stimuli.
Extracts time and frequency domain features.
Trains classification models (e.g., SVM) to distinguish between emotional states in the valence-arousal space.

# 📊 Motivation
Understanding the relationship between neural activity and emotional states can:
- Advance research in affective computing and brain-computer interfaces.
- Enable real-time emotion detection in HCI and therapeutic applications.
- Provide insights into neural correlates of emotions for psychology/neuroscience.

# 📁 Dataset
This project uses the DEAP dataset — a widely-used benchmark for EEG-based emotion research. It includes:
- EEG signals from 32 participants.
- Each participant watched 40 one-minute music video clips.
- Each trial includes self-reported scores for valence, arousal, dominance, liking, and familiarity. 


Important: DEAP is not included in this repo — please download it separately and place it in the appropriate data directory.
https://www.eecs.qmul.ac.uk/mmv/datasets/deap/

# 🧠 Project Workflow
1. Data Preprocessing
- Filtering, artifact removal, and segmentation of EEG signals. 
2. Feature Extraction & Selection
- Time-domain and frequency-domain features (e.g., band powers). 
3. Modeling
- Training classifiers such as SVM for emotion prediction. 
4. Evaluation
- Performance assessment using accuracy on valence/arousal classification. 

## 📂 Repository Structure

```text
Emotion-Analysis-using-EEG-signals/
│
├── FA(one vs all).py                  # Classifier for one-vs-all
├── FeatureALL(one vs all).py          # Feature-based classifier
├── PCA_windows.py                     # PCA + feature windowing
├── Feature Extraction & Selection.ipynb
├── Filtered EEG Signal_S01_Trail-10.ipynb
└── README.md                          # You’re here
```

## ⚙️ Usage

### 1️⃣ Clone the repository
```bash
git clone https://github.com/KhushhaliSharmaa/Emotion-Analysis-using-EEG-signals.git
cd Emotion-Analysis-using-EEG-signals
```

### 2️⃣ Prepare the dataset
- Download the **DEAP dataset** (EEG recordings + labels)
- Place the data in a folder (e.g., `data/DEAP/`)

### 3️⃣ Run preprocessing & feature extraction
Use the provided notebooks or scripts:
```bash
# Example
python PCA_windows.py
```

### 4️⃣ Train & Evaluate Models
Train classifiers like SVM and evaluate performance:
```bash
python "FA(one vs all).py"
```

> Replace script names as needed based on your experiment setup.

## 🛠 Dependencies

You’ll need the following common data-science libraries:

```bash
pip install numpy scipy scikit-learn pandas matplotlib mne
```

These dependencies are standard for EEG-based machine learning workflows using the DEAP dataset.

## 📈 Results & Evaluation

Model performance was evaluated on EEG data from **32 subjects** using a one-vs-all classification strategy based on the **valence–arousal model**.

### Average Classification Accuracy (%)

| Features       | HAHV | LAHV | HALV | LALV |
|----------------|------|------|------|------|
| FA (Beta Band) | 50.00 | 52.50 | 48.44 | 47.81 |
| All Features   | 58.70 | 79.35 | 64.83 | 72.58 |

**Key insights:**
- Using **all features** consistently outperforms beta-band-only features.
- Best performance is observed for the **LAHV** emotional state.
- Beta-band features alone yield near-chance accuracy, highlighting the importance of **multi-feature representations**.

> Accuracies are averaged across 32 subjects.

### ⚠️ Limitations
- Performance may vary across subjects due to EEG inter-subject variability.
- The model was evaluated on a fixed dataset (DEAP) without cross-dataset validation.

### 🔮 Future Work
- Explore deep learning models (CNNs, LSTMs) for emotion classification.
- Perform cross-subject and cross-dataset evaluation.
- Investigate real-time EEG emotion recognition.



