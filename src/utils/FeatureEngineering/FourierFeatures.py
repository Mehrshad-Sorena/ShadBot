from .Frequencies import Frequencies
from .DatasetIO import DatasetIO
from progress.bar import Bar
from .Config import Config
import pandas as pd
import numpy as np

#Functions:

#FreqReader()
#FeaturePerparer()
#Run()
#Get()

#/////////////////////

class FourierFeatures():

	def FreqReader(self, dataset, symbol, mode, number_frequencies):

		frequences = Frequencies()
		frequences = frequences.Get(dataset = dataset, symbol = symbol, mode = mode, number_frequencies = number_frequencies)
		
		return frequences

	
	def FeaturesPerparer(self, dataset, column, frequencies, symbol):

		fourier_feature = pd.DataFrame()
		features = {}

		for freq in frequencies['freq_' + column].dropna():

			time = np.arange(len(dataset.index), dtype=np.float32)
			k = 2 * np.pi * (1 / freq) * time
			
			features.update({
					        f"sin_{int(freq)}_{column}": np.sin(k),
					        f"cos_{int(freq)}_{column}": np.cos(k),
		    				})
		features = pd.DataFrame(features, index = dataset.index)

		return features

	
	def Run(self, dataset, symbol, mode, number_frequencies):

		frequencies = self.FreqReader(dataset = dataset, symbol = symbol, mode = mode, number_frequencies = number_frequencies)

		if (
			frequencies.empty == True and
			mode == None
			):

			frequencies = self.FreqReader(dataset = dataset, symbol = symbol, mode = 'Run')

		bar_config = Config()
		if bar_config.cfg['show_bar']:
			bar = Bar(symbol + ' ' + 'Fourier Features Finding: ', max = int(len(dataset.columns)))

		fourier_feature = pd.DataFrame(np.zeros(len(dataset.index)))

		for column in dataset.columns:

			if 'time' in column: continue

			feature_prepared = self.FeaturesPerparer(
														dataset = dataset,
														column = column,
														frequencies = frequencies,
														symbol = symbol
													)

			fourier_feature = fourier_feature.join(feature_prepared, how = 'right')

			if bar_config.cfg['show_bar']:
				bar.next()
		
		fourier_feature = fourier_feature.drop(columns = [0])

		datasetio = DatasetIO()
		datasetio.Write(name = 'fourier', dataset = fourier_feature, symbol = symbol)

		return fourier_feature


	def Get(self, dataset, symbol, mode, number_frequencies):

		datasetio = DatasetIO()

		if mode == 'Run':
			datasetio.Delete(symbol = symbol, name = 'fourier')
			return self.Run(dataset = dataset, symbol = symbol, mode = mode, number_frequencies = number_frequencies)

		elif mode == None:

			fourier_feature = datasetio.Read(name = 'fourier', symbol = symbol)
			
			if fourier_feature.empty == False:
				return fourier_feature

			else:
				return self.Run(dataset = dataset, symbol = symbol, mode = None, number_frequencies = number_frequencies)