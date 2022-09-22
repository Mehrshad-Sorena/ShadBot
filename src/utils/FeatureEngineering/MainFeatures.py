from src.indicators.StochAstic.StochAstic import StochAstic
from src.utils.Optimizers import NoiseCanceller
from .ParameterReader import ParameterReader
from src.indicators.MACD.MACD import MACD
from src.indicators.RSI.RSI import RSI
from .DatasetIO import DatasetIO
from progress.bar import Bar
from .Config import Config
import pandas_ta as ind
import pandas as pd
import numpy as np

#Functions:

#DatasetCreation()
#AlphaFactorOsilators()

#/////////////////


class MainFeatures():

	def __init__(self):

		#RSI Osilator Config:
		self.config_rsi_5m = True
		self.config_rsi_1h = True
		#///////////////////////

		#SMA Parameters:
		self.config_sma_5m = [True, True, True, True, True, True, True]
		self.sma_5m_length = [  14,   25,   50,  100,  150,  200,  250]

		self.config_sma_1h = [True, True, True, True, True, True, True]
		self.sma_1h_length = [  14,   25,   50,  100,  150,  200,  250]
		#/////////////////////////////////////


		#EMA Parameters:
		self.config_ema_5m = [True, True, True, True, True, True, True]
		self.ema_5m_length = [  14,   25,   50,  100,  150,  200,  250]

		self.config_ema_1h = [True, True, True, True, True, True, True]
		self.ema_1h_length = [  14,   25,   50,  100,  150,  200,  250]
		#/////////////////////////////////////

		#BBAND Parameters:
		self.config_bband_5m = True
		self.bband_5m_length = 5 
		self.bband_5m_std = 2 
		self.bband_5m_ddof = 0 
		self.bband_5m_mamod = 'sma'

		self.config_bband_1h = True
		self.bband_1h_length = 5 
		self.bband_1h_std = 2 
		self.bband_1h_ddof = 0 
		self.bband_1h_mamod = 'sma'
		#/////////////////////////////////////

		#IchiMokou Parameters:
		self.config_ichi_5m = True
		self.ichi_5m_tenkan = 9
		self.ichi_5m_kijun = 26
		self.ichi_5m_senkou = 52

		self.config_ichi_1h = True
		self.ichi_1h_tenkan = 9
		self.ichi_1h_kijun = 26
		self.ichi_1h_senkou = 52
		#/////////////////////////////////////

	#Main Dataset Creation:
	def DatasetCreation(self, dataset_5M, dataset_1H):

		dataset_5m = dataset_5M.copy(deep = True)
		dataset_5m.index = dataset_5m['time']

		dataset = pd.DataFrame()
		dataset = dataset.assign(
								close_5m = dataset_5m['close'],
								open_5m = dataset_5m['open'],
								low_5m = dataset_5m['low'],
								high_5m = dataset_5m['high'],
								HL2_5m = dataset_5m['HL/2'],
								HLC3_5m = dataset_5m['HLC/3'],
								HLCC4_5m = dataset_5m['HLCC/4'],
								OHLC4_5m = dataset_5m['OHLC/4'],
								volume_5m = dataset_5m['volume'],
								time_5m = dataset_5m['time'],
								)

		dataset.index = dataset['time_5m']

		dataset_1h = dataset_1H.copy(deep = True)
		dataset_1h.index = dataset_1h['time']

		dataset = dataset.assign(
								close_1h = dataset_1h['close'],
								open_1h = dataset_1h['open'],
								low_1h = dataset_1h['low'],
								high_1h = dataset_1h['high'],
								HL2_1h = dataset_1h['HL/2'],
								HLC3_1h = dataset_1h['HLC/3'],
								HLCC4_1h = dataset_1h['HLCC/4'],
								OHLC4_1h = dataset_1h['OHLC/4'],
								volume_1h = dataset_1h['volume'],
								time_1h = dataset_1h['time'],
								)

		# dataset.index = range(0 , len(dataset['close_5m']))
		return dataset 
	#/////////////////////////////////


	#Noise Filterd Feature:
	def AlphaFactorNoiseFilter(self, dataset):

		noise_canceller = NoiseCanceller.NoiseCanceller()

		dataset['time'] = dataset.index
		dataset.index = range(0, len(dataset.index))

		dataset = dataset.assign(
								close_5m_filter = noise_canceller.NoiseWavelet(dataset = dataset.copy(deep = True), applyto = 'close_5m'),
								open_5m_filter = noise_canceller.NoiseWavelet(dataset = dataset.copy(deep = True), applyto = 'open_5m'),
								high_5m_filter = noise_canceller.NoiseWavelet(dataset = dataset.copy(deep = True), applyto = 'high_5m'),
								low_5m_filter = noise_canceller.NoiseWavelet(dataset = dataset.copy(deep = True), applyto = 'low_5m'),
								HL2_5m_filter = noise_canceller.NoiseWavelet(dataset = dataset.copy(deep = True), applyto = 'HL2_5m'),
								HLC3_5m_filter = noise_canceller.NoiseWavelet(dataset = dataset.copy(deep = True), applyto = 'HLC3_5m'),
								HLCC4_5m_filter = noise_canceller.NoiseWavelet(dataset = dataset.copy(deep = True), applyto = 'HLCC4_5m'),
								OHLC4_5m_filter = noise_canceller.NoiseWavelet(dataset = dataset.copy(deep = True), applyto = 'OHLC4_5m'),

								close_1h_filter = noise_canceller.NoiseWavelet(dataset = dataset.copy(deep = True), applyto = 'close_1h'),
								open_1h_filter = noise_canceller.NoiseWavelet(dataset = dataset.copy(deep = True), applyto = 'open_1h'),
								high_1h_filter = noise_canceller.NoiseWavelet(dataset = dataset.copy(deep = True), applyto = 'high_1h'),
								low_1h_filter = noise_canceller.NoiseWavelet(dataset = dataset.copy(deep = True), applyto = 'low_1h'),
								HL2_1h_filter = noise_canceller.NoiseWavelet(dataset = dataset.copy(deep = True), applyto = 'HL2_1h'),
								HLC3_1h_filter = noise_canceller.NoiseWavelet(dataset = dataset.copy(deep = True), applyto = 'HLC3_1h'),
								HLCC4_1h_filter = noise_canceller.NoiseWavelet(dataset = dataset.copy(deep = True), applyto = 'HLCC4_1h'),
								OHLC4_1h_filter = noise_canceller.NoiseWavelet(dataset = dataset.copy(deep = True), applyto = 'OHLC4_1h'),
							)

		dataset.index = dataset['time']
		dataset = dataset.drop(columns = ['time'])

		return dataset

	#/////////////////////////////////


	def AlphaFactorOsilators(self, dataset, dataset_5M, dataset_1H, symbol):

		signalpriority = ['primary', 'secondry', 'primary', 'secondry']
		signaltype = ['buy' , 'sell', 'sell' , 'buy']
		timeframes = ['5M' , '1H']
		indicator_names = ['macd', 'stochastic', 'rsi']

		parameter_reader = ParameterReader()

		bar_config = Config()
		if bar_config.cfg['show_bar']:
			bar = Bar(
						symbol + ' ' + 'Alpha Factor Osilators Finding: ', 
																	max = int(
																			len(signalpriority) * 
																			len(signaltype) * 
																			len(timeframes) * 
																			len(indicator_names)
																			)
					)

		for ind_name in indicator_names:
			for timfrm in timeframes:
				for sigpriority, sigtype in zip(signalpriority, signaltype):

					# dataset['pattern_' + ind_name + '_div_' + timfrm + '_' + sigtype + '_' + sigpriority] = np.nan
					if timfrm == '1H':
						ind_parameters, ind_config, div_parameters, div_config = parameter_reader.Divergence(
																											signalpriority = sigpriority,
																											signaltype = sigtype,
																											symbol = symbol,
																											timeframe = timfrm,
																											dataset = dataset_1H,
																											indicator_name = ind_name
																											)
					elif timfrm == '5M':
						ind_parameters, ind_config, div_parameters, div_config = parameter_reader.Divergence(
																											signalpriority = sigpriority,
																											signaltype = sigtype,
																											symbol = symbol,
																											timeframe = timfrm,
																											dataset = dataset_5M,
																											indicator_name = ind_name
																											)
					#Add MACD Calculate Params AS Alpha Factor To Dataset:
					if ind_name == 'macd':

						ind = MACD(parameters = ind_parameters, config = ind_config)
						ind_calc = ind.calculator_macd()

						if timfrm == '5M':
							ind_calc['time'] = dataset_5M[symbol]['time']
							ind_calc['index'] = ind_calc.index
							ind_calc.index = ind_calc['time']

						elif timfrm == '1H':
							ind_calc['time'] = dataset_1H[symbol]['time']
							ind_calc['index'] = ind_calc.index
							ind_calc.index = ind_calc['time']

						column_div = ind_parameters.elements['MACD_column_div']

						dataset[column_div + '_' + timfrm.lower() + '_' + sigtype + '_' + sigpriority] = np.nan
						dataset[column_div + '_' + timfrm.lower() + '_' + sigtype + '_' + sigpriority] = ind_calc[column_div]

						ind_calc.index = ind_calc['index']
					#///////////////////////////////////////

					#Add StochAstic Calculate Params AS Alpha Factor To Dataset:
					elif ind_name == 'stochastic':

						ind = StochAstic(parameters = ind_parameters, config = ind_config)
						ind_calc = ind.calculator_StochAstic()

						if timfrm == '5M':
							ind_calc['time'] = dataset_5M[symbol]['time']
							ind_calc['index'] = ind_calc.index
							ind_calc.index = ind_calc['time']

						elif timfrm == '1H':
							ind_calc['time'] = dataset_1H[symbol]['time']
							ind_calc['index'] = ind_calc.index
							ind_calc.index = ind_calc['time']

						column_div = ind_parameters.elements['StochAstic_column_div']

						dataset[column_div + '_' + timfrm.lower() + '_' + sigtype + '_' + sigpriority] = np.nan
						dataset[column_div + '_' + timfrm.lower() + '_' + sigtype + '_' + sigpriority] = ind_calc[column_div]

						ind_calc.index = ind_calc['index']
					#//////////////////////////////////////////////////////////

					#Add RSI Calculate Params AS Alpha Factor To Dataset:
					elif ind_name == 'rsi':

						dataset['rsi_' + timfrm.lower() + '_' + sigtype + '_' + sigpriority] = np.nan

						ind = RSI(parameters = ind_parameters, config = ind_config)
						ind_calc = ind.calculator_rsi()

						if timfrm == '5M':
							ind_calc['time'] = dataset_5M[symbol]['time']
							ind_calc['index'] = ind_calc.index
							ind_calc.index = ind_calc['time']

						elif timfrm == '1H':
							ind_calc['time'] = dataset_1H[symbol]['time']
							ind_calc['index'] = ind_calc.index
							ind_calc.index = ind_calc['time']

						dataset['rsi_' + timfrm.lower() + '_' + sigtype + '_' + sigpriority] = ind_calc['rsi']

						ind_calc.index = ind_calc['index']
					#////////////////////////////////////////////////////////////
					if bar_config.cfg['show_bar']:
						bar.next()
		return dataset

	#Trend Factors:
	def AlphaFactorBBAND(self, dataset):

		if self.config_bband_5m == True:
			bband_ind_5m = ind.bbands(
									dataset['close_5m'], 
									length = self.bband_5m_length, #5, 
									std = self.bband_5m_std, #2, 
									ddof = self.bband_5m_ddof, #0, 
									mamod = self.bband_5m_mamod #'sma'
									)

			bband_lower_5m = bband_ind_5m[bband_ind_5m.columns[0]]
			bband_mid_5m = bband_ind_5m[bband_ind_5m.columns[1]]
			bband_upper_5m = bband_ind_5m[bband_ind_5m.columns[2]]
			# bband_bandwidth_5m = bband_ind_5m[bband_ind_5m.columns[3]]
			# bband_percent_5m = bband_ind_5m[bband_ind_5m.columns[4]]

			dataset = dataset.assign(
									bband_lower_5m = bband_lower_5m,
									bband_mid_5m = bband_mid_5m,
									bband_upper_5m = bband_upper_5m,
									# bband_bandwidth_5m = bband_bandwidth_5m,
									# bband_percent_5m = bband_percent_5m,
									)


		if self.config_bband_1h == True:
			bband_ind_1h = ind.bbands(
									dataset['close_1h'].dropna(), 
									length = self.bband_1h_length, #5, 
									std = self.bband_1h_std, #2, 
									ddof = self.bband_1h_ddof, #0, 
									mamod = self.bband_1h_mamod #'sma'
									)

			bband_lower_1h = bband_ind_1h[bband_ind_1h.columns[0]]
			bband_mid_1h = bband_ind_1h[bband_ind_1h.columns[1]]
			bband_upper_1h = bband_ind_1h[bband_ind_1h.columns[2]]
			bband_bandwidth_1h = bband_ind_1h[bband_ind_1h.columns[3]]
			bband_percent_1h = bband_ind_1h[bband_ind_1h.columns[4]]

			dataset = dataset.assign(
									bband_lower_1h = bband_lower_1h,
									bband_mid_1h = bband_mid_1h,
									bband_upper_1h = bband_upper_1h,
									# bband_bandwidth_1h = bband_bandwidth_1h,
									# bband_percent_1h = bband_percent_1h,
									)

		return dataset

	def AlphaFactorSMA(self, dataset):

		bar_config = Config()
		if bar_config.cfg['show_bar']:
			bar = Bar(
						symbol + ' ' + 'Alpha Factor SMA 5M Finding: ', 
						max = int(len(self.config_sma_5m))
					)

		counter = 0
		for elm in self.config_sma_5m:

			if elm == True:
				sma_ind_5m = ind.sma(dataset['close_5m'], length = self.sma_5m_length[counter])

				dataset[f'sma_5m_{self.sma_5m_length[counter]}'] = sma_ind_5m

			counter += 1

			if bar_config.cfg['show_bar']: bar.next()


		bar_config = Config()
		if bar_config.cfg['show_bar']:
			bar = Bar(
						symbol + ' ' + 'Alpha Factor SMA 1H Finding: ', 
						max = int(len(self.config_sma_1h))
					)

		counter = 0
		for elm in self.config_sma_1h:

			if elm == True:
				sma_ind_1h = ind.sma(dataset['close_1h'].dropna(), length = self.sma_1h_length[counter])

				dataset[f'sma_1h_{self.sma_1h_length[counter]}'] = sma_ind_1h

			counter += 1

			if bar_config.cfg['show_bar']: bar.next()
		
		return dataset

	def AlphaFactorEMA(self, dataset):

		bar_config = Config()
		if bar_config.cfg['show_bar']:
			bar = Bar(
						symbol + ' ' + 'Alpha Factor EMA 5M Finding: ', 
						max = int(len(self.config_ema_5m))
					)

		counter = 0
		for elm in self.config_ema_5m:

			if elm == True:

				ema_ind_5m = ind.ema(dataset['close_5m'], length = self.ema_5m_length[counter])

				dataset[f'ema_5m_{self.ema_5m_length[counter]}'] = ema_ind_5m

			counter += 1

			if bar_config.cfg['show_bar']: bar.next()


		bar_config = Config()
		if bar_config.cfg['show_bar']:
			bar = Bar(
						symbol + ' ' + 'Alpha Factor EMA 1H Finding: ', 
						max = int(len(self.config_ema_1h))
					)

		counter = 0
		for elm in self.config_ema_1h:

			if elm == True:

				ema_ind_1h = ind.ema(dataset['close_1h'].dropna(), length = self.ema_1h_length[counter])

				dataset[f'ema_1h_{self.ema_1h_length[counter]}'] = ema_ind_1h

			counter += 1

			if bar_config.cfg['show_bar']: bar.next()

		return dataset

	def AlphaFactorIchimokou(self, dataset):

		if self.config_ichi_5m == True:

			ichi_ind_5m, _ = ind.ichimoku(
										high = dataset['high_5m'],
										low = dataset['low_5m'],
										close = dataset['close_5m'],
										tenkan = self.ichi_5m_tenkan,
										kijun = self.ichi_5m_kijun,
										senkou = self.ichi_5m_senkou
										)

			spana_5m = ichi_ind_5m[ichi_ind_5m.columns[0]]
			spanb_5m = ichi_ind_5m[ichi_ind_5m.columns[1]]
			tenkan_5m = ichi_ind_5m[ichi_ind_5m.columns[2]]
			kijun_5m = ichi_ind_5m[ichi_ind_5m.columns[3]]
			chikou_5m = ichi_ind_5m[ichi_ind_5m.columns[4]]

			dataset = dataset.assign(
									spana_5m = spana_5m,
									spanb_5m = spanb_5m,
									tenkan_5m = tenkan_5m,
									kijun_5m = kijun_5m,
									chikou_5m = chikou_5m,
									)

		if self.config_ichi_1h == True:

			ichi_ind_1h, _ = ind.ichimoku(
										high = dataset['high_1h'].dropna(),
										low = dataset['low_1h'].dropna(),
										close = dataset['close_1h'].dropna(),
										tenkan = self.ichi_1h_tenkan,
										kijun = self.ichi_1h_kijun,
										senkou = self.ichi_1h_senkou
										)

			spana_1h = ichi_ind_1h[ichi_ind_1h.columns[0]]
			spanb_1h = ichi_ind_1h[ichi_ind_1h.columns[1]]
			tenkan_1h = ichi_ind_1h[ichi_ind_1h.columns[2]]
			kijun_1h = ichi_ind_1h[ichi_ind_1h.columns[3]]
			chikou_1h = ichi_ind_1h[ichi_ind_1h.columns[4]]

			dataset = dataset.assign(
									spana_1h = spana_1h,
									spanb_1h = spanb_1h,
									tenkan_1h = tenkan_1h,
									kijun_1h = kijun_1h,
									chikou_1h = chikou_1h,
									)
		

		return dataset
	#//////////////////////////////////////////////////////


	def Run(self, dataset_5M, dataset_1H, symbol):
		
		dataset = self.DatasetCreation(dataset_5M = dataset_5M[symbol], dataset_1H = dataset_1H[symbol])

		dataset = self.AlphaFactorNoiseFilter(dataset = dataset)

		dataset = self.AlphaFactorOsilators(
										dataset = dataset, 
										dataset_5M = dataset_5M, 
										dataset_1H = dataset_1H,
										symbol = symbol
										)

		dataset = self.AlphaFactorBBAND(dataset = dataset)
		dataset = self.AlphaFactorSMA(dataset = dataset)
		dataset = self.AlphaFactorEMA(dataset = dataset)
		dataset = self.AlphaFactorIchimokou(dataset = dataset)

		datasetio = DatasetIO()
		datasetio.Write(name = 'main', dataset = dataset, symbol = symbol)

		return dataset


	def Get(self, dataset_5M, dataset_1H, symbol, mode):
		
		datasetio = DatasetIO()

		if mode == None:
			dataset = datasetio.Read(name = 'main', symbol = symbol)
			dataset = dataset.drop(columns = ['time'])

			if dataset.empty == False:
				return dataset

			else:
				return self.Run(
								symbol = symbol, 
								dataset_5M = dataset_5M, 
								dataset_1H = dataset_1H
								)

		if mode == 'Run':

			datasetio.Delete(name = 'main', symbol = symbol)

			return self.Run(
							symbol = symbol, 
							dataset_5M = dataset_5M, 
							dataset_1H = dataset_1H
							)
