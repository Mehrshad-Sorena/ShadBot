from pykalman import KalmanFilter
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pywt

class NoiseCanceller:

	def __init__(self):

		#Noise Wavelet:
		self.scale_haar = 0.00008
		self.scale_db6 = 0.00008
		self.scale_dmey = 0.00128
		#///////////////////////////


	def NoiseKalmanFilter(self, dataset, applyto):

		kf = KalmanFilter(
						transition_matrices = [1],
						observation_matrices = [1],
						initial_state_mean = 0,
						initial_state_covariance = 1,
						observation_covariance=1,
						transition_covariance=.01,
						em_vars=['transition_covariance', 'observation_covariance']
						)
		# for elm in dataset.columns:
		# 	print(elm)

		kf = kf.em(dataset[applyto].dropna(), n_iter=5)

		(filtered_state_means, filtered_state_covariances) = kf.filter(dataset[applyto].dropna())
		(smoothed_state_means, smoothed_state_covariances) = kf.smooth(filtered_state_means)

		# state_means, _ = kf.filter(dataset[applyto].dropna())
		dataset_output = dataset.copy(deep = True)
		dataset_output['KalmanSmoothed' + applyto] = np.nan
		smoothed_state_means = pd.DataFrame(smoothed_state_means, columns = [applyto])
		smoothed_state_means = smoothed_state_means.reindex(index = dataset[applyto].index).shift((len(dataset[applyto].index) - len(dataset[applyto].dropna().index)))
		dataset_output['KalmanSmoothed' + applyto] = smoothed_state_means
		
		dataset_output[applyto] = dataset_output['KalmanSmoothed' + applyto]
		dataset_output = dataset_output.drop(columns = ['KalmanSmoothed' + applyto])

		# plt.plot(dataset[applyto].index[50:-1], dataset[applyto][50:-1], c = 'r')
		#plt.plot(dataset[applyto].index[250:-1], filtered_state_means[250:-1], c = 'b')
		# plt.plot(dataset[applyto].index[250:-1], smoothed_state_means[250:-1], c = 'g')
		# plt.show()

		return dataset_output[applyto]

	def NoiseWavelet(self, dataset, applyto):

		pywt.families(short=True)
		['Haar', 'Daubechies', 
		'Symlets', 'Coiflets', 
		'Biorthogonal', 'Reverse biorthogonal', 
		'Discrete Meyer (FIR Approximation)', 'Gaussian', 
		'Mexican hat wavelet', 'Morlet wavelet', 
		'Complex Gaussian wavelets', 'Shannon wavelets', 
		'Frequency B-Spline wavelets', 'Complex Morlet wavelets']

		# print(dataset[applyto].dropna())

		wavelet = "haar"
		coefficients = pywt.wavedec(dataset[applyto].dropna(), wavelet, mode='per')
		coefficients[1:] = [pywt.threshold(i, value = self.scale_haar*dataset[applyto].mean(), mode='soft') for i in coefficients[1:]]
		reconstructed_signal_haar = pywt.waverec(coefficients, wavelet, mode='per')

		wavelet = "db6"
		coefficients = pywt.wavedec(dataset[applyto].dropna(), wavelet, mode='per')
		coefficients[1:] = [pywt.threshold(i, value = self.scale_db6*dataset[applyto].mean(), mode='soft') for i in coefficients[1:]]
		reconstructed_signal_db6 = pywt.waverec(coefficients, wavelet, mode='per')


		wavelet = "dmey"
		coefficients = pywt.wavedec(dataset[applyto].dropna(), wavelet, mode='per')
		coefficients[1:] = [pywt.threshold(i, value = self.scale_dmey*dataset[applyto].mean(), mode='soft') for i in coefficients[1:]]
		reconstructed_signal_dmey = pywt.waverec(coefficients, wavelet, mode='per')

		reconstructed_signal = (reconstructed_signal_haar * reconstructed_signal_dmey * reconstructed_signal_db6) ** (1/3)

		# dataset[applyto].plot(color="b", alpha=0.5, label='original signal', lw=2, 
		# title=f'Threshold Scale: {scale:.1f}')

		# pd.Series(reconstructed_signal_haar, index = range((len(dataset[applyto].index) - len(dataset[applyto].dropna().index))-1 , len(dataset[applyto].index))).plot(c='r', 
		# label='DWT smoothing}', linewidth=1)
		# pd.Series(reconstructed_signal_db6, index = range((len(dataset[applyto].index) - len(dataset[applyto].dropna().index))-1 , len(dataset[applyto].index))).plot(c='orange', 
		# label='DWT smoothing}', linewidth=1)
		# pd.Series(reconstructed_signal_dmey, index = range((len(dataset[applyto].index) - len(dataset[applyto].dropna().index))-1 , len(dataset[applyto].index))).plot(c='k', 
		# label='DWT smoothing}', linewidth=1)
		# pd.Series(reconstructed_signal, index = range((len(dataset[applyto].index) - len(dataset[applyto].dropna().index))-1 , len(dataset[applyto].index))).plot(c='g', 
		# label='DWT smoothing}', linewidth=1)

		# plt.show()

		dataset['WaveletSmoothed_' + applyto] = np.nan

		reconstructed_signal = pd.DataFrame(reconstructed_signal, columns = [applyto])
		reconstructed_signal = reconstructed_signal.reindex(index = dataset[applyto].index).shift((len(dataset[applyto].index) - len(dataset[applyto].dropna().index)))
		
		dataset['WaveletSmoothed_' + applyto] = reconstructed_signal
		dataset[applyto] = dataset['WaveletSmoothed_' + applyto].values
		dataset = dataset.drop(columns = ['WaveletSmoothed_' + applyto])

		for clm in dataset.columns:
			if 'WaveletSmoothed_' in clm:
				dataset = dataset.drop(columns = [clm])

		return dataset[applyto]