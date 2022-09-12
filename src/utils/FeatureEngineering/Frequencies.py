from .DatasetIO import DatasetIO
from src.utils.Optimizers.Optimizers import Optimizers
from src.utils.Optimizers import NoiseCanceller
from progress.bar import Bar
import pandas as pd
import numpy as np


#Functions:

#Finder()
#Run()
#Get

#/////////////


class Frequencies():

	def Finder(self, dataset, column, period_format, period_coef):

		optimizer = Optimizers()
		noise_canceller = NoiseCanceller.NoiseCanceller()

		freq_time = str(period_coef) + period_format
		freq = [np.nan, np.nan, np.nan, np.nan, np.nan]
		freq_out = 1
		dataset_filterd = dataset.copy(deep = True)
		dataset_filterd['freq'] = np.nan

		for i in range(0, 5):

			if freq_out == 0: freq_out = 1

			noise_canceller.scale_haar = 1/freq_out
			noise_canceller.scale_db6 = 1/freq_out
			noise_canceller.scale_dmey = 1/freq_out
			
			dataset_filterd['freq'] = noise_canceller.NoiseWavelet(
																	dataset = dataset.reset_index(), 
																	applyto = column
																	).values

			dataset_freq = dataset_filterd.resample(freq_time).last()

			freq_out, frequences, spectrum = optimizer.FreqFinder(ts = dataset_freq['freq'].dropna())

			if freq_out == 0: break
			freq_time = str(period_coef * freq_out) + period_format
			freq[i] = freq_out * period_coef

		return freq


	def Run(self, dataset, symbol):
		
		frequencies = pd.DataFrame(np.zeros(5))

		bar = Bar(symbol + ' ' + 'Frequencies Finding: ', max = int(len(dataset.columns)))

		for data_column in dataset.columns:

			if 'time' in data_column: continue

			if '5m' in data_column:
				period_format = 'T'
				period_coef = 5

			elif '1h' in data_column:
				period_format = 'H'
				period_coef = 1
			
			frequencies['freq_' + data_column] = np.nan

			frequencies['freq_' + data_column] = (
													self.Finder(
															dataset = dataset.drop(
																					columns = ['time_5m', 'time_1h']
																					)
																					.copy(deep = True),
															column = data_column, 
															period_format = period_format, 
															period_coef = period_coef
															)
													)

			bar.next()

		frequencies = frequencies.drop(columns = [0])

		datasetio = DatasetIO()
		datasetio.Write(name = 'frequency', dataset = frequencies, symbol = symbol)

		return frequencies

	def Get(self, dataset, symbol, mode):

		datasetio = DatasetIO()
		frequencies = datasetio.Read(name = 'frequency', symbol = symbol)

		if mode == 'Run':
			datasetio.Delete(symbol = symbol, name = 'frequency')
			return self.Run(dataset = dataset, symbol = symbol)

		elif mode == None:

			if frequencies.empty == False:
				return frequencies

			else:
				return self.Run(dataset = dataset, symbol = symbol)