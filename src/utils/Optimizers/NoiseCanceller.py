from pykalman import KalmanFilter
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pywt

class NoiseCanceller:

	def __init__(self):

		pass


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

		dataset['KalmanSmoothed' + applyto] = np.nan
		smoothed_state_means = pd.DataFrame(smoothed_state_means, columns = [applyto])
		smoothed_state_means = smoothed_state_means.reindex(index = dataset[applyto].index).shift((len(dataset[applyto].index) - len(dataset[applyto].dropna().index)))
		dataset['KalmanSmoothed' + applyto] = smoothed_state_means
		dataset[applyto] = dataset['KalmanSmoothed' + applyto]
		dataset = dataset.drop(columns = ['KalmanSmoothed' + applyto])

		# plt.plot(dataset[applyto].index[50:-1], dataset[applyto][50:-1], c = 'r')
		#plt.plot(dataset[applyto].index[250:-1], filtered_state_means[250:-1], c = 'b')
		# plt.plot(dataset[applyto].index[250:-1], smoothed_state_means[250:-1], c = 'g')
		# plt.show()

		return dataset[applyto]

	def NoiseWavelet(self, dataset, applyto):

		pywt.families(short=True)
		['Haar', 'Daubechies', 
		'Symlets', 'Coiflets', 
		'Biorthogonal', 'Reverse biorthogonal', 
		'Discrete Meyer (FIR Approximation)', 'Gaussian', 
		'Mexican hat wavelet', 'Morlet wavelet', 
		'Complex Gaussian wavelets', 'Shannon wavelets', 
		'Frequency B-Spline wavelets', 'Complex Morlet wavelets']

		wavelet = "haar"
		scale = 0.00008
		coefficients = pywt.wavedec(dataset[applyto].dropna(), wavelet, mode='per')
		coefficients[1:] = [pywt.threshold(i, value = scale*dataset[applyto].mean(), mode='soft') for i in coefficients[1:]]
		reconstructed_signal_haar = pywt.waverec(coefficients, wavelet, mode='per')

		wavelet = "db6"
		scale = 0.00008
		coefficients = pywt.wavedec(dataset[applyto].dropna(), wavelet, mode='per')
		coefficients[1:] = [pywt.threshold(i, value = scale*dataset[applyto].mean(), mode='soft') for i in coefficients[1:]]
		reconstructed_signal_db6 = pywt.waverec(coefficients, wavelet, mode='per')


		wavelet = "dmey"#dmey
		scale = 0.00128
		coefficients = pywt.wavedec(dataset[applyto].dropna(), wavelet, mode='per')
		coefficients[1:] = [pywt.threshold(i, value = scale*dataset[applyto].mean(), mode='soft') for i in coefficients[1:]]
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

		dataset['WaveletSmoothed' + applyto] = np.nan

		reconstructed_signal = pd.DataFrame(reconstructed_signal, columns = [applyto])
		reconstructed_signal = reconstructed_signal.reindex(index = dataset[applyto].index).shift((len(dataset[applyto].index) - len(dataset[applyto].dropna().index)))
		
		dataset['WaveletSmoothed' + applyto] = reconstructed_signal
		dataset[applyto] = dataset['WaveletSmoothed' + applyto]
		dataset = dataset.drop(columns = ['WaveletSmoothed' + applyto])

		return dataset[applyto]