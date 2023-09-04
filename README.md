# Analysis of Neural Correlates of Emotion: An EEG-Based Approach
This repository contains the code and documentation for an analysis of neural correlates of emotion using an EEG-based approach. The primary objective of this study is to design an offline classification algorithm capable of classifying four quadrants of the valence-arousal plane and to report the accuracy of these classifications. The study aims to explore the interaction between EEG signals and different emotional classes by leveraging the valence-arousal theory of emotion. The proposed emotional recognition system utilizes EEG signals from 32 subjects, collected from the DEAP dataset, to classify different emotional classes.

# Dataset
The EEG data used in this study is sourced from the DEAP dataset. It comprises EEG recordings from 32 subjects while they were exposed to emotional stimuli. The dataset includes information on various emotional classes, allowing us to explore neural correlates of emotion using EEG signals.

# Project Overview
This study can be broken down into three key components:

# 1. Data Collection and Pre-processing
The EEG signals from the DEAP dataset are collected and pre-processed to prepare them for feature extraction and classification. Pre-processing may involve tasks such as filtering, artifact removal, and segmentation.

# 2. Feature Extraction and Selection
This study conducts an in-depth examination of various EEG features extracted from all 32 EEG channels. Feature selection techniques are employed to identify the most relevant features for emotion classification. The choice of features is crucial for achieving accurate classifications.

# 3. Classification Model
The feature matrices are labeled according to subjective emotional classes, and a classification model is created for each emotion class separately. Machine learning algorithms, such as Support Vector Machines (SVM), are employed to train the classification models. The models are evaluated based on their accuracy in classifying emotional states on the valence-arousal plane.

# Implementation
The project is implemented using the Python programming language.

Machine learning algorithms, specifically SVM, are utilized to build the classification models.

Data pre-processing, feature extraction, and feature selection are implemented using libraries like NumPy, SciPy, and scikit-learn.

# Acknowledgments
The DEAP dataset creators and contributors for providing valuable EEG data for research.

The open-source community for developing libraries and tools that facilitate the analysis of EEG signals and emotions.

The researchers and scientists in the field of emotion recognition using EEG signals, whose work has contributed to the foundation of this study.




