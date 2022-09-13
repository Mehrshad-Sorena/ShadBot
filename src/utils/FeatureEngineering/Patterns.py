from src.indicators.StochAstic.StochAstic import StochAstic
from src.utils.Divergence.Divergence import Divergence
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

#CandlePatterns()

#Run()
#Get()

#////////////////////


class Patterns():

	def __init__(self):

		self.config_candle_pattern_5m = True
		self.config_candle_pattern_1h = True

		#Run Flags:
		self.CandlePatternFlag = True
		self.DailyPatternFlag = True
		self.DivergencePatternFlag = True
		#///////////////////////////


	#Candle Patterns:
	def CandlePatterns(self, dataset, symbol):

		candle_patterns = pd.DataFrame()

		if self.config_candle_pattern_5m == True:

			cdl_patterns_5m = ind.cdl_pattern(
											open_ = dataset['open_5m'],
											high = dataset['high_5m'],
											low = dataset['low_5m'],
											close = dataset['close_5m'],
											name = 'all'
											)

			bar_config = Config()
			if bar_config.cfg['show_bar']:
				bar = Bar(symbol + ' ' + 'Candle Patterns 5M Finding: ', max = int(len(cdl_patterns_5m.columns)))

			counter = 0
			for pattern in cdl_patterns_5m.columns:

				candle_patterns[f'pattern_5m_{counter}'] = cdl_patterns_5m[pattern]

				candle_patterns[f'pattern_5m_{counter}'] = candle_patterns[f'pattern_5m_{counter}'].apply(
																	lambda x: 'buy' if x == 100 else ('sell' if x == -100 else np.nan)
																	)
				if bar_config.cfg['show_bar']:
					bar.next()

				counter += 1

		if self.config_candle_pattern_1h == True:

			cdl_patterns_1h = ind.cdl_pattern(
											open_ = dataset['open_1h'].dropna(),
											high = dataset['high_1h'].dropna(),
											low = dataset['low_1h'].dropna(),
											close = dataset['close_1h'].dropna(),
											name = 'all'
											)

			if bar_config.cfg['show_bar']:
				bar = Bar(symbol + ' ' + 'Candle Patterns 1H Finding: ', max = int(len(cdl_patterns_5m.columns)))

			counter = 0
			for pattern in cdl_patterns_1h.columns:

				candle_patterns[f'pattern_1h_{counter}'] = cdl_patterns_1h[pattern]

				candle_patterns[f'pattern_1h_{counter}'] = candle_patterns[f'pattern_1h_{counter}'].apply(
																lambda x: 'buy' if x == 100 else ('sell' if x == -100 else np.nan)
																)

				if bar_config.cfg['show_bar']:
					bar.next()

				counter += 1

		return candle_patterns
	#////////////////////////////////////////


	#Daily Pattern:
	def DailyPatterns(self, dataset, symbol):

		DaysOfWeek = ['Monday', 'Tuesday', 'Thursday', 'Wednesday', 'Friday']

		bar_config = Config()
		if bar_config.cfg['show_bar']:
			bar = Bar(symbol + ' ' + 'Daily Patterns Finding: ', max = int(len(DaysOfWeek)))

		daily_pattern = pd.DataFrame()

		for day in DaysOfWeek:

			daily_pattern['pattern_day_' + day] = np.nan
			daily_pattern['pattern_day_' + day] = dataset['time_5m'].apply(lambda x: 1 if x.day_name() == day else np.nan)

			if bar_config.cfg['show_bar']:
				bar.next()

		return daily_pattern
	#//////////////////////


	#Divergence Pattern:
	def DivergencePatterns(self, dataset, dataset_5M, dataset_1H, symbol):

		signalpriority = ['primary', 'secondry', 'primary', 'secondry']
		signaltype = ['buy' , 'sell', 'sell' , 'buy']
		timeframes = ['5M' , '1H']
		indicator_names = ['macd', 'stochastic', 'rsi']

		parameter_reader = ParameterReader()

		divergence_pattern = pd.DataFrame(index = dataset.index)

		bar_config = Config()
		if bar_config.cfg['show_bar']:
			bar = Bar(
						symbol + ' ' + 'Daily Patterns Finding: ', 
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

					if timfrm == '1H' and not dataset_1H: continue
					if timfrm == '5M' and not dataset_5M: continue

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

						ind_calc.index = ind_calc['index']

						column_div = ind_parameters.elements['MACD_column_div']
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

						ind_calc.index = ind_calc['index']

						column_div = ind_parameters.elements['StochAstic_column_div']
					#//////////////////////////////////////////////////////////

					#Add RSI Calculate Params AS Alpha Factor To Dataset:
					elif ind_name == 'rsi':

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

						ind_calc.index = ind_calc['index']

						column_div = 'rsi'
					#////////////////////////////////////////////////////////////



					divergence = Divergence(parameters = div_parameters, config = div_config)

					if timfrm == '1H':
						signal, _, _ = divergence.divergence(
															sigtype = sigtype,
															sigpriority = sigpriority,
															indicator = ind_calc,
															column_div = column_div,
															ind_name = ind_name,
															dataset_5M = dataset_1H,
															dataset_1H = dataset_1H,
															symbol = symbol,
															flaglearn = False,
															flagtest = False
															)

						signal = signal.drop_duplicates(subset = ['time_low_front'])
						signal.index = signal['time_low_front']

						divergence_pattern['pattern_' + ind_name + '_div_' + '1h_' + sigtype + '_' + sigpriority] = signal['signal']


					elif timfrm == '5M':
						signal, _, _ = divergence.divergence(
															sigtype = sigtype,
															sigpriority = sigpriority,
															indicator = ind_calc,
															column_div = column_div,
															ind_name = ind_name,
															dataset_5M = dataset_5M,
															dataset_1H = dataset_5M,
															symbol = symbol,
															flaglearn = False,
															flagtest = False
															)

						signal = signal.drop_duplicates(subset = ['time_low_front'])
						signal.index = signal['time_low_front']

						divergence_pattern['pattern_' + ind_name + '_div_' + '5m_' + sigtype + '_' + sigpriority] = signal['signal']

					if bar_config.cfg['show_bar']:
						bar.next()

		return divergence_pattern
	#/////////////////////

	def Run(self, dataset, symbol, dataset_5M, dataset_1H):
		
		datasetio = DatasetIO()

		pattern = pd.DataFrame(index = dataset.index)

		if self.CandlePatternFlag == True:
			candle_pattern = self.CandlePatterns(dataset = dataset, symbol = symbol)
			pattern = pattern.join(candle_pattern, how = 'right')

		if self.DailyPatternFlag == True:
			daily_pattern = self.DailyPatterns(dataset = dataset, symbol = symbol)
			pattern = pattern.join(daily_pattern, how = 'right')

		if self.DivergencePatternFlag == True:
			divergence_pattern = self.DivergencePatterns(
														dataset = dataset, 
														dataset_5M = dataset_5M, 
														dataset_1H = dataset_1H, 
														symbol = symbol
														)
			pattern = pattern.join(divergence_pattern, how = 'right')

		datasetio.Write(name = 'pattern', dataset = pattern, symbol = symbol)

		return pattern


	def Get(self, dataset, symbol, mode = None, dataset_5M = {}, dataset_1H = {}):

		datasetio = DatasetIO()

		pattern = pd.DataFrame(index = dataset.index)
		
		if mode == None:
			pattern = datasetio.Read(name = 'pattern', symbol = symbol)

			if pattern.empty == False:
				return pattern

			else:
				return self.Run(
							dataset = dataset, 
							symbol = symbol, 
							dataset_5M = dataset_5M, 
							dataset_1H = dataset_1H
							)

		if mode == 'Run':
			datasetio.Delete(name = 'pattern', symbol = symbol)

			return self.Run(
							dataset = dataset, 
							symbol = symbol, 
							dataset_5M = dataset_5M, 
							dataset_1H = dataset_1H
							)
			
