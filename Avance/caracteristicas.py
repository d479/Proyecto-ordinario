# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 18:11:58 2021

UNIVERSIDAD AUTÓNOMA DEL ESTADO DE MÉXICO
CU UAEM ZUMPANGO
REDES NEURONALES

Tema:
Profesor: Dr. Asdrúbal López Chau
Descripción:

@author: david
"""
from pathlib import Path
from scipy.io import wavfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt, IPython.display as ipd
import librosa, librosa.display
import sklearn
#from pyAudioAnalysis import audioBasicIO
#import audioFeatureExtraction
import math

kick_signals = [
    librosa.load(p)[0] for p in Path().glob('./Hombre 1.wav')
]
snare_signals = [
    librosa.load(p)[0] for p in Path().glob('./Mujer1.wav')
]
kick_signals[0].dtype

ipd.Audio('./Hombre .wav')
len(kick_signals)
plt.figure(figsize=(15, 6))
for i, x in enumerate(kick_signals):
    plt.subplot(2, 5, i+1)
    librosa.display.waveplot(x[:10000])
    plt.ylim(-1, 1)
    plt.tight_layout()
    
def extract_features(signal):
    return [
        librosa.feature.zero_crossing_rate(signal)[0, 0], #Numero de cruces por cero
        librosa.feature.spectral_centroid(signal)[0, 0], # centroide del espectro
    ]
kick_features = np.array([extract_features(x) for x in kick_signals])
print(kick_features)
plt.figure(figsize=(14, 5))
plt.hist(kick_features[:,0], color='b', range=(0, 0.2), alpha=0.5, bins=20)
plt.legend(('kicks', 'snares'))
plt.title('Histograma de los Zero Crossing rate')
plt.xlabel('Zero Crossing Rate')
plt.ylabel('Cuenta');

feature_table = np.vstack((kick_features))
print(feature_table.shape)

scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))
training_features = scaler.fit_transform(feature_table)
print(training_features.min(axis=0))
print(training_features.max(axis=0))
plt.figure(figsize=(14, 5))
plt.scatter(training_features[:10,0], training_features[:10,1], c='b')
plt.scatter(training_features[10:,0], training_features[10:,1], c='r')
plt.legend(('kicks', 'snares'))
plt.title('Zero Crossing rate Vs Spectral centroid')
plt.xlabel('Zero Crossing Rate')
plt.ylabel('Spectral Centroid');

T = 3.0      # duracion en segundos
sr = 22050   # frecuencia de muestreo en Hertz
# La amplitud ser a creciente logaritmicamente
amplitude = np.logspace(-3, 0, int(T*sr), endpoint=False, base=10.0) # amplitud variable en el tiempo
print(amplitude.min(), amplitude.max())
t = np.linspace(0, T, int(T*sr), endpoint=False)
x = amplitude*np.sin(2*np.pi*440*t)
ipd.Audio(x, rate=sr)
plt.figure(figsize=(14, 5))
librosa.display.waveplot(x, sr=sr);
frame_length = 1024 #tamaño de muestras de los segmentos
hop_length = 512
def rmse(x):
    return np.sqrt(np.mean(x**2))
plt.figure(figsize=(14, 5))
plt.semilogy([rmse(x[i:i+frame_length])
              for i in range(0, len(x), hop_length)]);
plt.title('Energia Creciente Logaritmicamente');
plt.figure(figsize=(14, 5))
frames = librosa.util.frame(x, frame_length=frame_length, hop_length=hop_length)
plt.semilogy([rmse(frame) for frame in frames.T]);
plt.title('Energia Creciente Logaritmicamente');
x, sr = librosa.load('./Mujer1.wav')
print(sr)
x.shape
librosa.get_duration(x, sr)
ipd.Audio(x, rate=sr)
librosa.display.waveplot(x, sr=sr);
hop_length = 256 # tamaño del incremento
frame_length = 512 # tamaño del segmento
energy = np.array([
    sum(abs(x[i:i+frame_length]**2))
    for i in range(0, len(x), hop_length)
])
energy.shape
rmse = librosa.feature.rms(x, frame_length=frame_length, hop_length=hop_length, center=True)
rmse.shape
rmse = rmse[0]
frames = range(len(energy))
t = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
plt.figure(figsize=(14, 5))
librosa.display.waveplot(x, sr=sr, alpha=0.4)
plt.plot(t, energy/energy.max(), 'r--')             # normalizada para la visualizacion
plt.plot(t[:len(rmse)], rmse/rmse.max(), color='g') # normalizada para la visualizacion
plt.legend(('Energia', 'RMSE'));
x, sr = librosa.load('./Mujer1.wav')
ipd.Audio(x, rate=sr)
plt.figure(figsize=(14, 5))
librosa.display.waveplot(x, sr=sr);n0 = 6500
n1 = 7500
plt.figure(figsize=(14, 5))
plt.plot(x[n0:n1]);
zero_crossings = librosa.zero_crossings(x[n0:n1], pad=False)
zero_crossings.shape
print(sum(zero_crossings))
zcrs = librosa.feature.zero_crossing_rate(x)
print(zcrs.shape)
plt.figure(figsize=(14, 5))
plt.plot(zcrs[0]);
"""
[Fs, x] = audioBasicIO.readAudioFile("./Mujer1.wav");
F, f_names = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050*Fs, 0.025*Fs);
plt.subplot(2,1,1); plt.plot(F[0,:]); plt.xlabel('Frame no'); plt.ylabel(f_names[0]); 
plt.subplot(2,1,2); plt.plot(F[1,:]); plt.xlabel('Frame no'); plt.ylabel(f_names[1]);
plt.rcParams['figure.figsize'] = [12, 20]
for num,val in enumerate(f_names):
    plt.subplot(math.ceil(len(f_names)/3),3,num+1); plt.plot(F[num,:]) 
    plt.xlabel('Frame no')
    plt.title(val)
    plt.ylabel(val) 
plt.tight_layout();
"""