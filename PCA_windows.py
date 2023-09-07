# -*- coding: utf-8 -*-
"""
Created on Wed May 18 21:51:40 2022

@author: hp
"""

import pandas as pd  
import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sstats
import scipy
# from scipy import signal
from scipy.integrate import simps
# from scipy.signal import filtfilt, butter, lfilter, welch
import nolds
import antropy as ant
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
#%%
#LOADING DATA(s01)
with open("s32.dat", "rb") as file: f = pickle.load(file, encoding='latin1')
Ntrial = 40
data = f['data']
labels = f['labels']
CAR_all = []
Feature_trial = []
BP_Power = []
Feature_array = []
FAA_trial = []
PCA_window = []
Class1_Value, Class2_Value, Class3_Value, Class4_Value = [] , [] , [] , []
accuracy_C1_value, accuracy_C2_value, accuracy_C3_value, accuracy_C4_value = [] , [] , [] , []
#%%
#LOOP FOR TRIAL
for k in range(Ntrial):
    trial = data[k]
    Ctrial = trial - np.mean(trial, axis=0) #Common Average Referencing
    C_EEG_Trial =  Ctrial [0:32, 384:8064] #Seperating EEG and non-EEG channels/ 3sec baseline removal
    C_Baseline_EEG = Ctrial [0:32, 0:384]
    CAR_all.append(C_EEG_Trial)
    bp_theta_Value, bp_alpha_Value, bp_beta_Value , bp_gamma_Value  = [] , [] , [] , []
    Mean_Value, Stdev_Value, Skew_Value, Median_Value, Kurt_Value, Ent_Value, HFD_Value, Corr_Value = [] , [] , [] , [] , [], [], [], []
#%%
    #LOOP FOR EEG CHANNELS
    for i in range (np.shape(C_EEG_Trial[:,1])[0]):
    # Filtering
      order = 5
      sample_freq = 128
      cutoff_freq = 2
      sample_duration = 60
      no_of_samples = sample_freq * sample_duration
      time = np.linspace(0, sample_duration, no_of_samples, endpoint = False)
      normalized_cutoff = 2 * cutoff_freq / sample_freq
      b, a = scipy.signal.butter(order, normalized_cutoff, analog=False)
      filtered_signal = scipy.signal.lfilter(b, a, C_EEG_Trial[0:32,:], axis = 0)
      filt_window = np.hsplit(filtered_signal, 60)      
      
      #Compute PSD
      sf = 128
      time = np.arange(C_EEG_Trial.size)/ sf  
      win = 1 * sf #Define window lenght(1 second)
      freqs_filt, psd_filt = freqs, Psd = scipy.signal.welch(filtered_signal, fs=128.0, nperseg=win, axis=1)
      freq_filt_res = freqs[1] - freqs[0]
      
      #BANDPASS FILTER
      nyq = 0.5 * sample_freq
      
      #FOR THETA
      fmin_theta = 3
      fmax_theta = 7
      low_theta = fmin_theta / nyq
      high_theta = fmax_theta / nyq
      b_theta, a_theta = scipy.signal.butter(order, [low_theta, high_theta], btype='band', analog=False)
      filtered_theta = scipy.signal.lfilter(b_theta, a_theta, filtered_signal[0:32,:], axis = 0)
      filt_theta = np.hsplit(filtered_theta, 60)
     
      
      #FOR ALPHA
      fmin_alpha = 8
      fmax_alpha = 13
      low_alpha = fmin_alpha / nyq
      high_alpha = fmax_alpha / nyq
      b_alpha, a_alpha = scipy.signal.butter(order, [low_alpha, high_alpha], btype='band', analog=False)
      filtered_alpha = scipy.signal.lfilter(b_alpha, a_alpha, filtered_signal[0:32,:], axis = 0)
      filt_alpha = np.hsplit(filtered_alpha, 60)
      
      
      #FOR BETA
      fmin_beta = 14
      fmax_beta = 29
      low_beta = fmin_beta / nyq
      high_beta = fmax_beta / nyq
      b_beta, a_beta = scipy.signal.butter(order, [low_beta, high_beta], btype='band', analog='False')
      filtered_beta = scipy.signal.lfilter(b_beta, a_beta, C_EEG_Trial[0:32,:], axis = 0)
      filt_beta = np.hsplit(filtered_beta, 60)
      #FOR Frontal assymetery of beta band
      freqs_b, Psd_b = scipy.signal.welch(filtered_beta, nperseg=win, axis=1)
      idx_b = np.logical_and(freqs_filt >= fmin_beta, freqs_filt <= fmax_beta)
      pow_rightbeta = simps(Psd_b[19,:][idx_b], dx=freq_filt_res)
      pow_leftbeta = simps(Psd_b[2,:][idx_b], dx=freq_filt_res)
      #Computing FAA at F3 & C4 channel
      FAA_beta = np.log(pow_rightbeta) - np.log(pow_leftbeta)

      #FOR GAMMA
      fmin_gamma = 30
      fmax_gamma = 47
      low_gamma = fmin_gamma / nyq
      high_gamma = fmax_gamma / nyq
      b_gamma, a_gamma = scipy.signal.butter(order, [low_gamma, high_gamma], btype='band', analog='False')
      filtered_gamma = scipy.signal.lfilter(b_gamma, a_gamma, C_EEG_Trial[0:32,:], axis = 0)
      filt_gamma = np.hsplit(filtered_gamma, 60)
#%%
      #Loop for 60 chunks
      #Compute PSD, bandpower
      for j in range (60):
          freqs, Psd = scipy.signal.welch(((filt_window[j])[i]),fs=128, nperseg = win, axis=0 )
          freqs_theta, Psd_theta = scipy.signal.welch(((filt_theta[j])[i]), fs=128, nperseg = win, axis=0)
          freqs_alpha, Psd_alpha = scipy.signal.welch(((filt_alpha[j])[i]), fs=128, nperseg = win, axis=0)
          freqs_beta, Psd_beta = scipy.signal.welch(((filt_beta[j])[i]), fs=128, nperseg = win, axis=0)
          freqs_gamma, Psd_gamma = scipy.signal.welch(((filt_gamma[j])[i]), fs=128, nperseg = win, axis=0)
          freqs_res = freqs[1] - freqs[0]
          idx_theta = np.logical_and(freqs >= fmin_theta, freqs <= fmax_theta)
          idx_alpha = np.logical_and(freqs >= fmin_alpha, freqs <= fmax_alpha)
          idx_beta = np.logical_and(freqs >= fmin_beta, freqs <= fmax_beta)
          idx_gamma = np.logical_and(freqs >= fmin_gamma, freqs <= fmax_gamma)
          bp_theta = simps(Psd_theta[idx_theta], dx=freqs_res)
          bp_alpha = simps(Psd_alpha[idx_alpha], dx=freqs_res)
          bp_beta = simps(Psd_beta[idx_beta], dx=freqs_res)
          bp_gamma = simps(Psd_gamma[idx_gamma], dx=freqs_res)
          bp_theta_Value.append(bp_theta)
          bp_alpha_Value.append(bp_alpha)
          bp_beta_Value.append(bp_beta)
          bp_gamma_Value.append(bp_gamma)
          
    n = len(filt_window)
    for l in range (n):
        for m in range (len(filt_window[l])):
            w_mean = np.mean((filt_window[l])[m])
            w_stdev = np.std((filt_window[l])[m])
            w_skew = sstats.skew((filt_window[l])[m])
            w_median = np.median((filt_window[l])[m])
            w_kurt = sstats.kurtosis((filt_window[l])[m])
            ent = ant.sample_entropy((filt_window[l])[m])
            hfd = ant.higuchi_fd((filt_window[l])[m])
            corr_dim = nolds.corr_dim(((filt_window[l])[m]), emb_dim=2)
            Mean_Value.append(w_mean)
            Stdev_Value.append(w_stdev)
            Skew_Value.append(w_skew)
            Median_Value.append(w_median)
            Kurt_Value.append(w_kurt)
            Ent_Value.append(ent)
            HFD_Value.append(hfd)
            Corr_Value.append(corr_dim)
    FAA_trial.append(FAA_beta)
    # frontal_assym = np.array(FAA_trial)
    Feature_array.append([Skew_Value, Kurt_Value, Ent_Value, HFD_Value, Corr_Value, bp_gamma_Value, bp_theta_Value, bp_alpha_Value, bp_beta_Value])
    Feature = np.array(Feature_array)
    Feature_matrix = np.reshape(Feature, (Feature.shape[0], Feature.shape[1]*Feature.shape[2]))
    # Feature_window = np.hsplit(Feature_reshape, 60)
    # Feature_matrix = np.column_stack((Feature_reshape, frontal_assym))
    # df_Feature = pd.DataFrame(Feature_matrix)
    df_Feature = pd.DataFrame(Feature_matrix)
    df_Feature[df_Feature==np.inf]=np.nan
    df_Feature.fillna(df_Feature.mean(), inplace=True)
    scaler=StandardScaler()
    Feature_rescaled = scaler.fit_transform(Feature_matrix)
    Feature_window = np.hsplit(Feature_rescaled, 60)
x = len(Feature_window)
for p in range(x):
    pca = PCA(n_components = 0.95)
    pca.fit(Feature_window[p])
    PCA_Feature = pca.transform(Feature_window[p])
    PCA_window.append(PCA_Feature)
#%%
HAHV_Class1 = [1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1] 
LAHV_Class2 = [0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
HALV_Class3 = [0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,1,1,1,1,1,0]
LALV_Class4 = [0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,1,0,0,0,0,0,0,0]
z = len(PCA_window)
for q in range(z):
    Class_1 = np.column_stack((PCA_window[q], HAHV_Class1))
    Class_2 = np.column_stack((PCA_window[q], LAHV_Class2))
    Class_3 = np.column_stack((PCA_window[q], HALV_Class3))
    Class_4 = np.column_stack((PCA_window[q], LALV_Class4))
    Class1_Value.append(Class_1)
    Class2_Value.append(Class_2)
    Class3_Value.append(Class_3)
    Class4_Value.append(Class_4)
#%%
#CLASSIFICATION HAHV/CLASS-1
y1_predict_Value = [] 
N = len(Class1_Value) 
for r in range(N):
    X1 = Class1_Value[r][:,0:-1]
    y1 = Class1_Value[r][:,-1]
    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.25, random_state=42)
    model = SVC(gamma='scale', random_state=42, class_weight='balanced')
    C1 = model.fit(X1_train, y1_train)
    accuracy_C1 = model.score(X1_test, y1_test)
    y1_predict = model.predict(X1_test)
    accuracy_C1_value.append(accuracy_C1)
    y1_predict_Value.append(y1_predict)
    
#CLASSIFICATION LAHV/CLASS-2
y2_predict_Value = [] 
K = len(Class2_Value)
for s in range(K):
    X2 = Class2_Value[s][:,0:-1]
    y2 = Class2_Value[s][:,-1]
    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.25, random_state=42)
    C2 = model.fit(X2_train, y2_train)
    accuracy_C2 = model.score(X2_test, y2_test)
    y2_predict = model.predict(X2_test)
    accuracy_C2_value.append(accuracy_C2)
    y2_predict_Value.append(y2_predict)

#CLASSIFICATION HALV/CLASS-3
y3_predict_Value = [] 
L = len(Class3_Value)
for t in range(L):
    X3 = Class3_Value[t][:,0:-1]
    y3 = Class3_Value[t][:,-1]
    X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.25, random_state=42)
    C3 = model.fit(X3_train, y3_train)
    accuracy_C3 = model.score(X3_test, y3_test)
    y3_predict = model.predict(X3_test)
    accuracy_C3_value.append(accuracy_C3)
    y3_predict_Value.append(y3_predict)
    
#CLASSIFICATION LALV/CLASS-4
y4_predict_Value = [] 
M = len(Class4_Value)
for w in range(M):
    X4 = Class4_Value[w][:,0:-1]
    y4 = Class4_Value[w][:,-1]
    X4_train, X4_test, y4_train, y4_test = train_test_split(X4, y4, test_size=0.25, random_state=42)
    C4 = model.fit(X4_train, y4_train)
    accuracy_C4 = model.score(X4_test, y4_test)
    y4_predict = model.predict(X4_test)
    accuracy_C4_value.append(accuracy_C4)
    y4_predict_Value.append(y4_predict)
#%%
#Plot Class1-HAHV
plt.plot(accuracy_C1_value, color='magenta', marker='o', mfc='blue')
plt.xlim(1,60)
plt.ylabel('Accuracy HAHV')
plt.xlabel('Time(s)')
plt.show()

# Plot Class2-LAHV
plt.plot(accuracy_C2_value, color='magenta', marker='o', mfc='blue')
plt.xlim(1,60)
plt.ylabel('Accuracy LAHV')
plt.xlabel('Time(s)')
plt.show()

#Plot Class3-HALV
plt.plot(accuracy_C3_value, color='magenta', marker='o', mfc='blue')
plt.xlim(1,60)
plt.ylabel('Accuracy HALV')
plt.xlabel('Time(s)')
plt.show()

# Plot Class4-LALV
plt.plot(accuracy_C4_value, color='magenta', marker='o', mfc='blue')
plt.xlim(1,60)
plt.ylabel('Accuracy LALV')
plt.xlabel('Time(s)')
plt.show()
#%%
    