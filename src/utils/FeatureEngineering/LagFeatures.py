from src.utils.FeatureEngineering.Frequencies import Frequencies
from statsmodels.tsa.stattools import acf
import pandas as pd

class LagFeatures():

	def __init__(self):

		self.lags = [1, 2, 3, 4, 5, 6, 7, 8, 9]

	def LagFinder(self, dataset, applyto):

		correlations, _ = acf(
							x = dataset[applyto].dropna(), 
							adjusted = False, 
							nlags = 500, 
							qstat = False, 
							fft = True, 
							alpha = 0.05
							)

		# print('corr = ', correlations)

		self.lags = []
		lag_counter = 0

		for corr in correlations:

			if len(self.lags) > 4: break

			if corr > 0.97:

				self.lags.append(lag_counter)
				if lag_counter == 0: self.lags.remove(0)

			lag_counter += 1

	def LagCreation(self, dataset, symbol):

		dataset_lag = dataset.copy(deep = True)
		dataset_lag.index = pd.to_datetime(dataset_lag['time_5m'])
		dataset_lag = dataset_lag.drop(columns = ['time_5m', 'time_1h'])

		outlier_cutoff = 0.01
		LagedData = pd.DataFrame()

		frequences = Frequencies()
		frequences = frequences.Get(dataset = dataset, symbol = symbol, mode = None, number_frequencies = 2)

		LagedData[f'real_{0}'] = (dataset_lag.copy(deep = True).stack())

		for freq_counter in range(0, 2):

			freq = frequences['freq_close_5m'][freq_counter]
			dataset_frequented = pd.DataFrame()
			dataset_frequented = dataset_lag.copy(deep = True).resample(str(freq) + 'T').last().dropna(subset=['close_5m'])
			self.LagFinder(dataset = dataset_frequented.reset_index().copy(deep = True), applyto = 'close_5m')

			for lag in self.lags:

				LagedData[f'return_{freq}_{lag}'] = (dataset_frequented
																	.pct_change(lag)
																	.stack()
																	# .pipe(lambda x:
																	# 				x.clip(
																	# 						lower=x.quantile(outlier_cutoff),
																	# 						upper=x.quantile(1-outlier_cutoff)
																	# 						)
																	# 	)
																	# .add(1)
																	# .pow(1/lag)
																	# .sub(1)
													)

			LagedData = self.MemontumCreation(dataset = LagedData, freq = freq)
			LagedData[f'real_{freq}'] = (dataset_frequented.stack())

		LagedData = LagedData.swaplevel()

		return LagedData


	def MemontumCreation(self, dataset, freq):

		for lag in self.lags:
			dataset[f'momentum_{freq}_{lag}'] = dataset[f'return_{freq}_{lag}'].sub(dataset[f'return_{freq}_{self.lags[0]}'])

		return dataset

	def LagShiftedCreation(self, dataset):

		for t in self.timelags:
			dataset[f'target_-{t}h'] = (dataset[f'return_{t}h'].shift(t))

		for t in self.lags:
			dataset[f'target_{t}h'] = (dataset[f'return_{t}h'].shift(-t))
		
		return dataset