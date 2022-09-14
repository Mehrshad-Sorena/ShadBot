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

			if len(self.lags) > 10: break

			if corr > 0.97:

				self.lags.append(lag_counter)
				if lag_counter == 0: self.lags.remove(0)

			lag_counter += 1

	def LagCreation(self, dataset):

		dataset_lag = dataset.copy(deep = True)
		dataset_lag.index = pd.to_datetime(dataset_lag['time_5m'])
		dataset_lag = dataset_lag.drop(columns = ['time_5m', 'time_1h'])

		dataset_frequented = dataset_lag.resample('20T').last()

		self.LagFinder(dataset = dataset_frequented.reset_index().copy(deep = True), applyto = 'close_5m')

		outlier_cutoff = 0.01
		LagedData = pd.DataFrame()

		for lag in self.lags:
			LagedData[f'return_{lag}'] = (dataset_frequented
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

		LagedData['real'] = (dataset_frequented.stack())

		LagedData = LagedData.swaplevel()

		return LagedData