from flask import Flask, request, render_template
import numpy as np
from scipy.fftpack import dct
import pickle
import librosa as lbr
import uuid
import matplotlib.pyplot as plt
from PIL import Image
import io
import os
import csv
from flask import Blueprint, request, jsonify
from http import HTTPStatus
import config.response_handler as ResponseHandler
from helpers.global_helper import GlobalHelper

model_path = 'test.pkl'
knn_model = pickle.load(open(model_path, 'rb'))

train_data = []
num_ceps = 12

with open('features.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        # Ambil fitur MFCC (kolom 1 hingga 11)
        mfcc_features = np.array([float(row[f'prefix_{i}']) for i in range(0, 12)])
        label = row['word']  # Ambil label dari kolom 'label'

        # Tambahkan data ke train_data dalam format yang sesuai
        train_data.append({'prefix': mfcc_features, 'word': label})

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'wav'}

def bacadata(mp):
  signal,sr  = lbr.load(mp,duration=30,sr=22050)
  return signal,sr

def initialize(mp):
  signal,sr = lbr.load(mp,duration=30,sr=22050)
  signal = signal[0:int(30 * sr)]
  return sr,signal

def lowPassFilter(signal, pre_emphasis=0.97):
	return np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

def preEmphasis(mp):
    sr, signal = initialize(mp)
    emphasizedSignal = lowPassFilter(signal)
    
   
    
   
   

    return emphasizedSignal
	

def framing(mp, frame_index=50):
    windowSize = 0.025
    windowStep = 0.01
    sr, signal = initialize(mp)
    frame_length, frame_step = windowSize * sr, windowStep * sr
    signal_length = len(preEmphasis(mp))
    overlap = int(round(frame_length))
    frameSize = int(round(frame_step))
    numberOfframes = int(np.ceil(float(np.abs(signal_length - frameSize)) / overlap ))
    pad_signal_length = numberOfframes * frameSize + overlap
    if pad_signal_length >= signal_length:
        z = np.zeros((pad_signal_length - signal_length))
        pad_signal = np.append(preEmphasis(mp), z)
    else:
        pad_signal = preEmphasis(mp)

    indices = np.tile(np.arange(0, overlap), (numberOfframes, 1)) + np.tile(np.arange(0,
                numberOfframes * frameSize, frameSize), (overlap, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    frames *= np.hamming(frame_length)

    # Menunjukkan hanya satu frame (misalnya, frame ke-100) dan hasil windowingnya
    frame_to_show = frames[frame_index]

 
   
    
    # Simpan gambar framing dan windowing

    return frames

def fouriertransform(mp):
	NFFT = 512
	frames = framing(mp)
	mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
	pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))
	return pow_frames

def filterbanks(mp):
  nfilt = 40
  low_freq_mel = 0
  NFFT = 512

  sr, signal = initialize(mp)
  high_freq_mel = (2595 * np.log10(1 + (sr / 2) / 700))
  mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)
  hz_points = (700 * (10**(mel_points / 2595) - 1))
  bin = np.floor((NFFT + 1) * hz_points / sr)

  pow_frames = fouriertransform(mp)
  fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
  for m in range(1, nfilt + 1):
      f_m_minus = int(bin[m - 1])
      f_m = int(bin[m])
      f_m_plus = int(bin[m + 1])

      for k in range(f_m_minus, f_m):
          fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
      for k in range(f_m, f_m_plus):
          fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
  filter_banks = np.dot(pow_frames, fbank.T)
  filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
  filter_banks = 20 * np.log10(filter_banks)


 


  return filter_banks
    
def mfcct(mp):
    cep_lifter = 22
    filter_banks = filterbanks(mp)

    # Hitung MFCC sebelum penggunaan cep_lifter
    mfcc_before = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)]
    (nframes, ncoeff) = mfcc_before.shape
    mfcc_before = np.mean(mfcc_before, axis=0)

  
    

    # Hitung MFCC setelah penggunaan cep_lifter
    mfcc_after = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)]

    (nframes, ncoeff) = mfcc_after.shape
    n = np.arange(ncoeff)
    lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
    mfcc_after *= lift
    mfcc_after = np.mean(mfcc_after, axis=0)

  
  

    return mfcc_before, mfcc_after

mfcc_blueprint = Blueprint('mfcc', __name__)

@mfcc_blueprint.route('/predict', methods=['POST'])
def predict():
    result = None
    distances = []
    if request.method == 'POST':
        file = request.files['file']
        unique_filename = str(uuid.uuid4()) + '.wav'
        file.save(unique_filename)
        mp = unique_filename
        mfcc_before, mfcc_after = mfcct(mp)
        mfcc_features = mfcc_after.reshape(1, -1)
        prediction = knn_model.predict(mfcc_features)
        filter_banks = filterbanks(mp)
        filterbank_table = filter_banks.tolist()

        # Hitung jarak Euclidean antara data baru dan data pelatihan
        for i, train_data_point in enumerate(train_data):
            # Ambil fitur MFCC dari train_data_point
            train_mfcc = train_data_point['prefix']
            
            # Misalnya, hitung jarak Euclidean antara mfcc_features dan train_mfcc
            distance = np.linalg.norm(mfcc_features - train_mfcc)
            distances.append((i, distance))
        # Urutkan jarak-jarak tersebut berdasarkan jarak
        distances.sort(key=lambda x: x[1])

        # Pilih 5 data terdekat
        k = 7
        nearest_neighbors = distances[:k]

        # Ambil kelas data terdekat
        predicted_classes = [train_data[i]['word'] for i, _ in nearest_neighbors]

        if predicted_classes[0] == '1':
            
            result = 'benda'
            return ResponseHandler.custom_success_response(status=HTTPStatus.OK, data={'data': {
                'result': result,
                'predicted_classes': predicted_classes,
                'euclidean_distances':[distance for _, distance in nearest_neighbors],
                
               
                }}, message="Sucses!")
            
          
            # dct_image= 'dct_plot.png'
            
        elif '2' in predicted_classes :
            result = 'benda2'
            return ResponseHandler.custom_success_response(status=HTTPStatus.OK, data={'data': {
           'result': result,
            'predicted_classes': predicted_classes,
            'euclidean_distances':[distance for _, distance in nearest_neighbors],
            
           
            }}, message="Sucses!")
            # dct_image= 'dct_plot.png'
            
        elif '3' in predicted_classes :
            result = 'kerja1'
            return ResponseHandler.custom_success_response(status=HTTPStatus.OK, data={'data': {
           'result': result,
            'predicted_classes': predicted_classes,
            'euclidean_distances':[distance for _, distance in nearest_neighbors],
            
           
            }}, message="Sucses!")
            # dct_image= 'dct_plot.png'
            
        elif '4' in predicted_classes :
            result = 'kerja2'
            return ResponseHandler.custom_success_response(status=HTTPStatus.OK, data={'data': {
            'result': result,
            'predicted_classes': predicted_classes,
            'euclidean_distances':[distance for _, distance in nearest_neighbors],
            
           
            }}, message="Sucses!")
            # dct_image= 'dct_plot.png'
            
        elif '5' in predicted_classes :
            result = 'sifat1'
            return ResponseHandler.custom_success_response(status=HTTPStatus.OK, data={'data': {
            'result': result,
            'predicted_classes': predicted_classes,
            'euclidean_distances':[distance for _, distance in nearest_neighbors],
            
           
            }}, message="Sucses!")
            # dct_image= 'dct_plot.png'
        elif '6' in predicted_classes :
            result = 'sifat2'
            return ResponseHandler.custom_success_response(status=HTTPStatus.OK, data={'data': {
           'result': result,
            'predicted_classes': predicted_classes,
            'euclidean_distances':[distance for _, distance in nearest_neighbors],
            
           
            }}, message="Sucses!")
            # dct_image= 'dct_plot.png'
        else :
            print(predicted_classes)
            return ResponseHandler.custom_success_response(status=HTTPStatus.BAD_REQUEST, data="", message="Invalid !")
        
        
        

      
       
       
       