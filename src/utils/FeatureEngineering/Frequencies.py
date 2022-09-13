from src.utils.Optimizers.Optimizers import Optimizers
from src.utils.Optimizers import NoiseCanceller
from .DatasetIO import DatasetIO
from progress.bar import Bar
from .Config import Config
import pandas as pd
import numpy as np


#Functions:

#Finder()
#Run()
#Get

#/////////////


class Frequencies():

	def Finder(self, dataset, column, period_format, period_coef, number_frequencies):

		optimizer = Optimizers()
		noise_canceller = NoiseCanceller.NoiseCanceller()

		freq_time = str(period_coef) + period_format
		freq = [np.nan] * number_frequencies
		freq_out = 1
		dataset_filterd = dataset.copy(deep = True)
		dataset_filterd['freq'] = np.nan

		for i in range(0, number_frequencies):

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


	def Run(self, dataset, symbol, number_frequencies):
		
		frequencies = pd.DataFrame(np.zeros(number_frequencies))

		bar_config = Config()
		if bar_config.cfg['show_bar']:
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
															period_coef = period_coef,
															number_frequencies = number_frequencies
															)
													)
			if bar_config.cfg['show_bar']:
				bar.next()

		frequencies = frequencies.drop(columns = [0])

		datasetio = DatasetIO()
		datasetio.Write(name = 'frequency', dataset = frequencies, symbol = symbol)

		return frequencies

	def Get(self, dataset, symbol, mode, number_frequencies):

		datasetio = DatasetIO()

		if mode == 'Run':
			datasetio.Delete(symbol = symbol, name = 'frequency')
			return self.Run(dataset = dataset, symbol = symbol, number_frequencies = number_frequencies)

		elif mode == None:

			frequencies = datasetio.Read(name = 'frequency', symbol = symbol)

			if frequencies.empty == False:
				return frequencies

			else:
				return self.Run(dataset = dataset, symbol = symbol, number_frequencies = number_frequencies)