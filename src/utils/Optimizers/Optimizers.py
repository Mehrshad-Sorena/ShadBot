import threading
from .Config import Config as ConfigOptimizers
from src.utils.Divergence.Parameters import Parameters as indicator_parameters
from src.utils.Divergence.Config import Config as indicator_config

from src.utils.Divergence.Divergence import Divergence

from src.indicators.MACD.Parameters import Parameters as MACDParameters
from src.indicators.MACD.Config import Config as MACDConfig
from src.indicators.MACD.MACD import MACD

from src.indicators.StochAstic.Parameters import Parameters as StochAsticParameters
from src.indicators.StochAstic.Config import Config as StochAsticConfig
from src.indicators.StochAstic.StochAstic import StochAstic

from src.indicators.RSI.Parameters import Parameters as RSIParameters
from src.indicators.RSI.Config import Config as RSIConfig
from src.indicators.RSI.RSI import RSI

from progress.bar import Bar
import pandas as pd
import numpy as np
import time
import os
import sys
import random
from random import randint


if 'win' in sys.platform:
	path_slash = '\\'
elif 'linux' in sys.platform:
	path_slash = '/'


class Optimizers():

	def __init__(self):

		self.symbol = 'XAUUSD_i'
		self.sigpriority = 'primary'
		self.sigtype = 'buy'
		self.main_turn = 10
		self.turn = 100
		self.dataset = pd.DataFrame()
		self.timeframe = '5M'


	def FreqFinder(self, ts, detrend='linear'):
	    from scipy.signal import periodogram

	    if self.timeframe == '5M':
	    	min_time = '5T'

	    elif self.timeframe == '1H':
	    	min_time = '1H'

	    fs = pd.Timedelta("1Y") / pd.Timedelta(min_time)

	    freqencies, spectrum = periodogram(
									        ts,
									        fs=fs,
									        detrend=detrend,
									        window="boxcar",
									        scaling='spectrum',
									    )
	    spectrum_counter = 0
	    for elm in spectrum:
	    	if elm == np.max(spectrum):
	    		freq = freqencies[spectrum_counter]
	    	spectrum_counter += 1

	    period_time = (freq * pd.Timedelta(min_time))/pd.Timedelta("1Y")

	    return round(freq), freqencies, spectrum


	def MacdOptimizer(self):

		print('Start MACD Optimizer ', self.sigtype, ' ', self.sigpriority, ' ', self.timeframe, ' ...')

		configoptimizers = ConfigOptimizers()
		macd_parameters = MACDParameters()
		macd_config = MACDConfig()

		ind_params = indicator_parameters()
		ind_config = indicator_config()

		self.dataset[self.symbol] = self.dataset[self.symbol].assign(index = self.dataset[self.symbol].index)

		freq, _, _ = self.FreqFinder(self.dataset[self.symbol].close)

		if self.timeframe == '5M':
			freq_time = str(5 * freq) + 'T'
		elif self.timeframe == '1H':
			freq_time = str(freq) + 'H'

		bar = Bar(self.sigtype + ' ' + self.sigpriority + ' ' + self.timeframe, max = int(self.turn))

		self.dataset[self.symbol] = self.dataset[self.symbol].set_index('time').resample(freq_time).last().dropna()
		self.dataset[self.symbol] = self.dataset[self.symbol].assign(time = self.dataset[self.symbol].index)
		self.dataset[self.symbol] = self.dataset[self.symbol].set_index('index')

		path = configoptimizers.cfg['path_MACD'] + path_slash + self.sigpriority + path_slash + self.sigtype + path_slash + self.timeframe + path_slash

		if not os.path.exists(path):
			os.makedirs(path)

		path = configoptimizers.cfg['path_MACD'] + path_slash + self.sigpriority + path_slash + self.sigtype + path_slash + self.timeframe + path_slash + self.symbol + '.csv'

		if os.path.exists(path):
			output_read = pd.read_csv(path).drop(columns = ['Unnamed: 0'])
		else:
			output_read = pd.DataFrame()
			output_read['MACD_apply_to'] = np.nan
			output_read['MACD_fast'] = np.nan
			output_read['MACD_slow'] = np.nan
			output_read['MACD_signal'] = np.nan
			output_read['MACD_column_div'] = np.nan
			output_read['corr_low'] = np.nan
			output_read['corr_high'] = np.nan
			output_read['Divergence_diff_extereme'] = np.nan
			output_read['Divergence_num_exteremes_min'] = np.nan
			output_read['Divergence_num_exteremes_max'] = np.nan
			output_read['frequency'] = np.nan
			output_read['score'] = np.nan
			
		
		output = pd.DataFrame(np.ones(self.turn))
		output['MACD_apply_to'] = np.nan
		output['MACD_fast'] = np.nan
		output['MACD_slow'] = np.nan
		output['MACD_signal'] = np.nan
		output['MACD_column_div'] = np.nan
		output['corr_low'] = np.nan
		output['corr_high'] = np.nan
		output['Divergence_diff_extereme'] = np.nan
		output['Divergence_num_exteremes_min'] = np.nan
		output['Divergence_num_exteremes_max'] = np.nan
		output['frequency'] = np.nan
		output['score'] = np.nan
		

		for i in range(self.turn):
			macd_parameters.elements['MACD' + '_apply_to'] = random.choice([
																		'open',
																		'close',
																		'low',
																		'high',
																		'HL/2',
																		'HLC/3',
																		'HLCC/4',
																		'OHLC/4'
																		])
			macd_parameters.elements['MACD' + '_fast'] = randint(2, 300)
			macd_parameters.elements['MACD' + '_slow'] = randint(2 , 700)
			macd_parameters.elements['MACD' + '_signal'] = randint(2 , 50)

			ind_params.elements['Divergence' + '_diff_extereme'] = randint(1 , 6)
			ind_params.elements['Divergence' + '_num_exteremes_min'] = randint(2 , 500)
			ind_params.elements['Divergence' + '_num_exteremes_max'] = randint(2 , 500)

			dive_column = random.choice(['macd', 'macds', 'macdh'])

			while macd_parameters.elements['MACD' + '_fast'] + 10 >= macd_parameters.elements['MACD' + '_slow']:
				macd_parameters.elements['MACD' + '_fast'] = randint(2, 300)
				macd_parameters.elements['MACD' + '_slow'] = randint(2 , 700)

			repeat_counter = 0
			if output.dropna().empty == False:

				repeat_checker_now = np.where(
											(macd_parameters.elements['MACD' + '_fast'] == output['MACD_fast'].values) &
											(macd_parameters.elements['MACD' + '_slow'] == output['MACD_slow'].values) &
											(macd_parameters.elements['MACD' + '_signal'] == output['MACD_signal'].values) &
											(ind_params.elements['Divergence' + '_diff_extereme'] == output['Divergence_diff_extereme'].values) &
											(ind_params.elements['Divergence' + '_num_exteremes_min'] == output['Divergence_num_exteremes_min'].values) &
											(ind_params.elements['Divergence' + '_num_exteremes_max'] == output['Divergence_num_exteremes_max'].values) &
											(macd_parameters.elements['MACD' + '_apply_to'] == output['MACD_apply_to'].values) &
											(dive_column == output['MACD_column_div'].values)
										)[0]

				repeat_checker_before = np.where(
											(macd_parameters.elements['MACD' + '_fast'] == output_read['MACD_fast'].values) &
											(macd_parameters.elements['MACD' + '_slow'] == output_read['MACD_slow'].values) &
											(macd_parameters.elements['MACD' + '_signal'] == output_read['MACD_signal'].values) &
											(ind_params.elements['Divergence' + '_diff_extereme'] == output_read['Divergence_diff_extereme'].values) &
											(ind_params.elements['Divergence' + '_num_exteremes_min'] == output_read['Divergence_num_exteremes_min'].values) &
											(ind_params.elements['Divergence' + '_num_exteremes_max'] == output_read['Divergence_num_exteremes_max'].values) &
											(macd_parameters.elements['MACD' + '_apply_to'] == output_read['MACD_apply_to'].values) &
											(dive_column == output_read['MACD_column_div'].values)
										)[0]

				while (
						len(repeat_checker_now) > 0 or
						len(repeat_checker_before) >0
						):
					macd_parameters.elements['MACD' + '_apply_to'] = random.choice([
																				'open',
																				'close',
																				'low',
																				'high',
																				'HL/2',
																				'HLC/3',
																				'HLCC/4',
																				'OHLC/4'
																				])
					macd_parameters.elements['MACD' + '_fast'] = randint(2, 300)
					macd_parameters.elements['MACD' + '_slow'] = randint(2 , 700)
					macd_parameters.elements['MACD' + '_signal'] = randint(2 , 50)

					ind_params.elements['Divergence' + '_diff_extereme'] = randint(1 , 6)
					ind_params.elements['Divergence' + '_num_exteremes_min'] = randint(2 , 250)
					ind_params.elements['Divergence' + '_num_exteremes_max'] = randint(2 , 250)

					dive_column = random.choice(['macd', 'macds', 'macdh'])

					while macd_parameters.elements['MACD' + '_fast'] + 10 >= macd_parameters.elements['MACD' + '_slow']:
						macd_parameters.elements['MACD' + '_fast'] = randint(2, 300)
						macd_parameters.elements['MACD' + '_slow'] = randint(2 , 700)

					repeat_checker_now = np.where(
											(macd_parameters.elements['MACD' + '_fast'] == output['MACD_fast'].values) &
											(macd_parameters.elements['MACD' + '_slow'] == output['MACD_slow'].values) &
											(macd_parameters.elements['MACD' + '_signal'] == output['MACD_signal'].values) &
											(ind_params.elements['Divergence' + '_diff_extereme'] == output['Divergence_diff_extereme'].values) &
											(ind_params.elements['Divergence' + '_num_exteremes_min'] == output['Divergence_num_exteremes_min'].values) &
											(ind_params.elements['Divergence' + '_num_exteremes_max'] == output['Divergence_num_exteremes_max'].values) &
											(macd_parameters.elements['MACD' + '_apply_to'] == output['MACD_apply_to'].values) &
											(dive_column == output['MACD_column_div'].values)
										)[0]

					repeat_checker_before = np.where(
												(macd_parameters.elements['MACD' + '_fast'] == output_read['MACD_fast'].values) &
												(macd_parameters.elements['MACD' + '_slow'] == output_read['MACD_slow'].values) &
												(macd_parameters.elements['MACD' + '_signal'] == output_read['MACD_signal'].values) &
												(ind_params.elements['Divergence' + '_diff_extereme'] == output_read['Divergence_diff_extereme'].values) &
												(ind_params.elements['Divergence' + '_num_exteremes_min'] == output_read['Divergence_num_exteremes_min'].values) &
												(ind_params.elements['Divergence' + '_num_exteremes_max'] == output_read['Divergence_num_exteremes_max'].values) &
												(macd_parameters.elements['MACD' + '_apply_to'] == output_read['MACD_apply_to'].values) &
												(dive_column == output_read['MACD_column_div'].values)
											)[0]

					if repeat_counter >= len(output_read['MACD_fast'].dropna().index): break
					repeat_counter += 1
				

			output['MACD_apply_to'][i] = macd_parameters.elements['MACD' + '_apply_to']
			output['MACD_fast'][i] = macd_parameters.elements['MACD' + '_fast']
			output['MACD_slow'][i] = macd_parameters.elements['MACD' + '_slow']
			output['MACD_signal'][i] = macd_parameters.elements['MACD' + '_signal']
			output['MACD_column_div'][i] = dive_column
			output['frequency'][i] = freq_time

			macd_parameters.elements['dataset_5M'] = self.dataset
			macd_parameters.elements['dataset_1H'] = self.dataset

			macd = MACD(parameters = macd_parameters, config = macd_config)
			macd_calc = macd.calculator_macd()


			macd = Divergence(parameters = ind_params, config = ind_config)
			signal, signaltype, indicator = macd.divergence(
															sigtype = self.sigtype,
															sigpriority = self.sigpriority,
															indicator = macd_calc,
															column_div = dive_column,
															ind_name = 'macd',
															dataset_5M = macd_parameters.elements['dataset_' + self.timeframe],
															dataset_1H = macd_parameters.elements['dataset_' + self.timeframe],
															symbol = self.symbol,
															flaglearn = False,
															flagtest = True
															)
			bar.next()

			if signal.empty == True: continue
			divergence_out = pd.DataFrame(np.ones(signal.index[-1]))
			divergence_out['macd'] = np.nan
			divergence_out['low'] = np.nan
			divergence_out['high'] = np.nan

			counter = 0
			for elm in signal.index:
				divergence_out['macd'][counter] = signal.indicator_front[elm]
				divergence_out['macd'][counter + 1] = signal.indicator_back[elm]

				divergence_out['low'][counter] = signal.low_front[elm]
				divergence_out['low'][counter + 1] = signal.low_back[elm]

				divergence_out['high'][counter] = signal.high_front[elm]
				divergence_out['high'][counter + 1] = signal.high_back[elm]

				counter += 2

			divergence_out = divergence_out.dropna()
			divergence_out = divergence_out.drop(columns = [0])

			number_divergence = len(divergence_out.index)/1000

			divergence_out = divergence_out.corr()

			output['score'][i] = -((divergence_out['macd'][2] * divergence_out['macd'][1] * number_divergence) ** (1/3))

			if (
				divergence_out['macd'][2] > 0 and
				divergence_out['macd'][1] > 0
				):
				output['score'][i] = -output['score'][i]

			output['corr_low'][i] = divergence_out['macd'][1]
			output['corr_high'][i] = divergence_out['macd'][2]
			output['Divergence_diff_extereme'][i] = ind_params.elements['Divergence' + '_diff_extereme']
			output['Divergence_num_exteremes_min'][i] = ind_params.elements['Divergence' + '_num_exteremes_min']
			output['Divergence_num_exteremes_max'][i] = ind_params.elements['Divergence' + '_num_exteremes_max']
			#print(output.head(i))
			#print('turn = ', self.main_turn * i, ', score = ', output_read['score'].min(), ' ', self.sigtype, ' ', self.sigpriority)

		if os.path.exists(path):
			os.remove(path)

		output = output.drop(columns = [0])
		output = pd.concat([output, output_read], ignore_index=True)

		output.dropna().sort_values(by = ['score'], ascending = False).to_csv(path)

		print()
		print('MACD Optimizer ', self.sigtype, ' ', self.sigpriority, ' ', self.timeframe, ' is Finished')

		return output.dropna().sort_values(by = ['score'], ascending = False)


	def StochAsticOptimizer(self):

		print('Start StochAstic Optimizer ', self.sigtype, ' ', self.sigpriority, ' ', self.timeframe, ' ...')

		configoptimizers = ConfigOptimizers()
		stochastic_parameters = StochAsticParameters()
		stochastic_config = StochAsticConfig()

		ind_params = indicator_parameters()
		ind_config = indicator_config()

		self.dataset[self.symbol] = self.dataset[self.symbol].assign(index = self.dataset[self.symbol].index)

		freq, _, _ = self.FreqFinder(self.dataset[self.symbol].close)

		if self.timeframe == '5M':
			freq_time = str(5 * freq) + 'T'
		elif self.timeframe == '1H':
			freq_time = str(freq) + 'H'

		bar = Bar(self.sigtype + ' ' + self.sigpriority + ' ' + self.timeframe, max = int(self.turn))

		self.dataset[self.symbol] = self.dataset[self.symbol].set_index('time').resample(freq_time).last().dropna()
		self.dataset[self.symbol] = self.dataset[self.symbol].assign(time = self.dataset[self.symbol].index)
		self.dataset[self.symbol] = self.dataset[self.symbol].set_index('index')

		path = configoptimizers.cfg['path_StochAstic'] + path_slash + self.sigpriority + path_slash + self.sigtype + path_slash + self.timeframe + path_slash

		if not os.path.exists(path):
			os.makedirs(path)

		path = configoptimizers.cfg['path_StochAstic'] + path_slash + self.sigpriority + path_slash + self.sigtype + path_slash + self.timeframe + path_slash + self.symbol + '.csv'

		if os.path.exists(path):
			output_read = pd.read_csv(path).drop(columns = ['Unnamed: 0'])
		else:
			output_read = pd.DataFrame()

			output_read['StochAstic_k'] = np.nan
			output_read['StochAstic_d'] = np.nan
			output_read['StochAstic_smooth_k'] = np.nan
			output_read['StochAstic_mamod'] = np.nan
			output_read['StochAstic_column_div'] = np.nan
			output_read['corr_low'] = np.nan
			output_read['corr_high'] = np.nan
			output_read['Divergence_diff_extereme'] = np.nan
			output_read['Divergence_num_exteremes_min'] = np.nan
			output_read['Divergence_num_exteremes_max'] = np.nan
			output_read['frequency'] = np.nan
			output_read['score'] = np.nan
			
		
		output = pd.DataFrame(np.ones(self.turn))
		output['StochAstic_k'] = np.nan
		output['StochAstic_d'] = np.nan
		output['StochAstic_smooth_k'] = np.nan
		output['StochAstic_mamod'] = np.nan
		output['StochAstic_column_div'] = np.nan
		output['corr_low'] = np.nan
		output['corr_high'] = np.nan
		output['Divergence_diff_extereme'] = np.nan
		output['Divergence_num_exteremes_min'] = np.nan
		output['Divergence_num_exteremes_max'] = np.nan
		output['frequency'] = np.nan
		output['score'] = np.nan
		

		for i in range(self.turn):
			stochastic_parameters.elements['StochAstic_mamod'] = random.choice([
																				'sma',
																				'ema',
																				'wma',
																				])
			stochastic_parameters.elements['StochAstic_k'] = randint(2, 500)
			stochastic_parameters.elements['StochAstic_d'] = randint(2 , 900)
			stochastic_parameters.elements['StochAstic_smooth_k'] = randint(2 , 50)

			ind_params.elements['Divergence' + '_diff_extereme'] = randint(1 , 6)
			ind_params.elements['Divergence' + '_num_exteremes_min'] = randint(2 , 500)
			ind_params.elements['Divergence' + '_num_exteremes_max'] = randint(2 , 500)

			#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


			dive_column = random.choice(['StochAstic_k', 'StochAstic_d'])

			while stochastic_parameters.elements['StochAstic_d'] + 2 >= stochastic_parameters.elements['StochAstic_k']:
				stochastic_parameters.elements['StochAstic_k'] = randint(2, 900)
				stochastic_parameters.elements['StochAstic_d'] = randint(2 , 900)

			repeat_counter = 0
			if output.dropna().empty == False:

				repeat_checker_now = np.where(
											(stochastic_parameters.elements['StochAstic_k'] == output['StochAstic_k'].values) &
											(stochastic_parameters.elements['StochAstic_d'] == output['StochAstic_d'].values) &
											(stochastic_parameters.elements['StochAstic_smooth_k'] == output['StochAstic_smooth_k'].values) &
											(ind_params.elements['Divergence' + '_diff_extereme'] == output['Divergence_diff_extereme'].values) &
											(ind_params.elements['Divergence' + '_num_exteremes_min'] == output['Divergence_num_exteremes_min'].values) &
											(ind_params.elements['Divergence' + '_num_exteremes_max'] == output['Divergence_num_exteremes_max'].values) &
											(stochastic_parameters.elements['StochAstic_mamod'] == output['StochAstic_mamod'].values) &
											(dive_column == output['StochAstic_column_div'].values)
										)[0]

				repeat_checker_before = np.where(
											(stochastic_parameters.elements['StochAstic_k'] == output_read['StochAstic_k'].values) &
											(stochastic_parameters.elements['StochAstic_d'] == output_read['StochAstic_d'].values) &
											(stochastic_parameters.elements['StochAstic_smooth_k'] == output_read['StochAstic_smooth_k'].values) &
											(ind_params.elements['Divergence' + '_diff_extereme'] == output_read['Divergence_diff_extereme'].values) &
											(ind_params.elements['Divergence' + '_num_exteremes_min'] == output_read['Divergence_num_exteremes_min'].values) &
											(ind_params.elements['Divergence' + '_num_exteremes_max'] == output_read['Divergence_num_exteremes_max'].values) &
											(stochastic_parameters.elements['StochAstic_mamod'] == output_read['StochAstic_mamod'].values) &
											(dive_column == output_read['StochAstic_column_div'].values)
										)[0]

				while (
						len(repeat_checker_now) > 0 or
						len(repeat_checker_before) >0
						):
					stochastic_parameters.elements['StochAstic_mamod'] = random.choice([
																						'sma',
																						'ema',
																						'wma',
																						])
					stochastic_parameters.elements['StochAstic_k'] = randint(2, 500)
					stochastic_parameters.elements['StochAstic_d'] = randint(2 , 900)
					stochastic_parameters.elements['StochAstic_smooth_k'] = randint(2 , 50)

					ind_params.elements['Divergence' + '_diff_extereme'] = randint(1 , 6)
					ind_params.elements['Divergence' + '_num_exteremes_min'] = randint(2 , 250)
					ind_params.elements['Divergence' + '_num_exteremes_max'] = randint(2 , 250)

					dive_column = random.choice(['macd', 'macds', 'macdh'])

					while stochastic_parameters.elements['StochAstic_d'] + 2 >= stochastic_parameters.elements['StochAstic_k']:
						stochastic_parameters.elements['StochAstic_k'] = randint(2, 700)
						stochastic_parameters.elements['StochAstic_d'] = randint(2 , 700)

					repeat_checker_now = np.where(
											(stochastic_parameters.elements['StochAstic_k'] == output['StochAstic_k'].values) &
											(stochastic_parameters.elements['StochAstic_d'] == output['StochAstic_d'].values) &
											(stochastic_parameters.elements['StochAstic_smooth_k'] == output['StochAstic_smooth_k'].values) &
											(ind_params.elements['Divergence' + '_diff_extereme'] == output['Divergence_diff_extereme'].values) &
											(ind_params.elements['Divergence' + '_num_exteremes_min'] == output['Divergence_num_exteremes_min'].values) &
											(ind_params.elements['Divergence' + '_num_exteremes_max'] == output['Divergence_num_exteremes_max'].values) &
											(stochastic_parameters.elements['StochAstic_mamod'] == output['StochAstic_mamod'].values) &
											(dive_column == output['StochAstic_column_div'].values)
										)[0]

					repeat_checker_before = np.where(
												(stochastic_parameters.elements['StochAstic_k'] == output_read['StochAstic_k'].values) &
												(stochastic_parameters.elements['StochAstic_d'] == output_read['StochAstic_d'].values) &
												(stochastic_parameters.elements['StochAstic_smooth_k'] == output_read['StochAstic_smooth_k'].values) &
												(ind_params.elements['Divergence' + '_diff_extereme'] == output_read['Divergence_diff_extereme'].values) &
												(ind_params.elements['Divergence' + '_num_exteremes_min'] == output_read['Divergence_num_exteremes_min'].values) &
												(ind_params.elements['Divergence' + '_num_exteremes_max'] == output_read['Divergence_num_exteremes_max'].values) &
												(stochastic_parameters.elements['StochAstic_mamod'] == output_read['StochAstic_mamod'].values) &
												(dive_column == output_read['StochAstic_column_div'].values)
											)[0]

					if repeat_counter >= len(output_read['StochAstic_k'].dropna().index): break
					repeat_counter += 1
				

			output['StochAstic_mamod'][i] = stochastic_parameters.elements['StochAstic_mamod']
			output['StochAstic_k'][i] = stochastic_parameters.elements['StochAstic_k']
			output['StochAstic_d'][i] = stochastic_parameters.elements['StochAstic_d']
			output['StochAstic_smooth_k'][i] = stochastic_parameters.elements['StochAstic_smooth_k']
			output['StochAstic_column_div'][i] = dive_column
			output['frequency'][i] = freq_time

			stochastic_parameters.elements['dataset_5M'] = self.dataset
			stochastic_parameters.elements['dataset_1H'] = self.dataset

			stochastic = StochAstic(parameters = stochastic_parameters, config = stochastic_config)
			stochastic_calc = stochastic.calculator_StochAstic()


			stochastic = Divergence(parameters = ind_params, config = ind_config)
			signal, signaltype, indicator = stochastic.divergence(
																sigtype = self.sigtype,
																sigpriority = self.sigpriority,
																indicator = stochastic_calc,
																column_div = dive_column,
																ind_name = 'stochastic',
																dataset_5M = stochastic_parameters.elements['dataset_' + self.timeframe],
																dataset_1H = stochastic_parameters.elements['dataset_' + self.timeframe],
																symbol = self.symbol,
																flaglearn = False,
																flagtest = True
																)
			bar.next()

			if signal.empty == True: continue
			divergence_out = pd.DataFrame(np.ones(signal.index[-1]))
			divergence_out['stochastic'] = np.nan
			divergence_out['low'] = np.nan
			divergence_out['high'] = np.nan

			counter = 0
			for elm in signal.index:
				divergence_out['stochastic'][counter] = signal.indicator_front[elm]
				divergence_out['stochastic'][counter + 1] = signal.indicator_back[elm]

				divergence_out['low'][counter] = signal.low_front[elm]
				divergence_out['low'][counter + 1] = signal.low_back[elm]

				divergence_out['high'][counter] = signal.high_front[elm]
				divergence_out['high'][counter + 1] = signal.high_back[elm]

				counter += 2

			divergence_out = divergence_out.dropna()
			divergence_out = divergence_out.drop(columns = [0])

			number_divergence = len(divergence_out.index)/1000

			divergence_out = divergence_out.corr()

			output['score'][i] = -((divergence_out['stochastic'][2] * divergence_out['stochastic'][1] * number_divergence) ** (1/3))

			if (
				divergence_out['stochastic'][2] > 0 and
				divergence_out['stochastic'][1] > 0
				):
				output['score'][i] = -output['score'][i]

			output['corr_low'][i] = divergence_out['stochastic'][1]
			output['corr_high'][i] = divergence_out['stochastic'][2]
			output['Divergence_diff_extereme'][i] = ind_params.elements['Divergence' + '_diff_extereme']
			output['Divergence_num_exteremes_min'][i] = ind_params.elements['Divergence' + '_num_exteremes_min']
			output['Divergence_num_exteremes_max'][i] = ind_params.elements['Divergence' + '_num_exteremes_max']
			#print(output.head(i))
			#print('turn = ', self.main_turn * i, ', score = ', output_read['score'].min(), ' ', self.sigtype, ' ', self.sigpriority)

		if os.path.exists(path):
			os.remove(path)

		output = output.drop(columns = [0])
		output = pd.concat([output, output_read], ignore_index=True)

		output.dropna().sort_values(by = ['score'], ascending = False).to_csv(path)

		print()
		print('StochAstic Optimizer ', self.sigtype, ' ', self.sigpriority, ' ', self.timeframe, ' is Finished')

		return output.dropna().sort_values(by = ['score'], ascending = False)


	def RSIOptimizer(self):

		print('Start RSI Optimizer ', self.sigtype, ' ', self.sigpriority, ' ', self.timeframe, ' ...')

		configoptimizers = ConfigOptimizers()
		rsi_parameters = RSIParameters()
		rsi_config = RSIConfig()

		ind_params = indicator_parameters()
		ind_config = indicator_config()

		self.dataset[self.symbol] = self.dataset[self.symbol].assign(index = self.dataset[self.symbol].index)

		freq, _, _ = self.FreqFinder(self.dataset[self.symbol].close)

		if self.timeframe == '5M':
			freq_time = str(5 * freq) + 'T'
		elif self.timeframe == '1H':
			freq_time = str(freq) + 'H'

		bar = Bar(self.sigtype + ' ' + self.sigpriority + ' ' + self.timeframe, max = int(self.turn))

		self.dataset[self.symbol] = self.dataset[self.symbol].set_index('time').resample(freq_time).last().dropna()
		self.dataset[self.symbol] = self.dataset[self.symbol].assign(time = self.dataset[self.symbol].index)
		self.dataset[self.symbol] = self.dataset[self.symbol].set_index('index')

		path = configoptimizers.cfg['path_RSI'] + path_slash + self.sigpriority + path_slash + self.sigtype + path_slash + self.timeframe + path_slash

		if not os.path.exists(path):
			os.makedirs(path)

		path = configoptimizers.cfg['path_RSI'] + path_slash + self.sigpriority + path_slash + self.sigtype + path_slash + self.timeframe + path_slash + self.symbol + '.csv'

		if os.path.exists(path):
			output_read = pd.read_csv(path).drop(columns = ['Unnamed: 0'])
		else:
			output_read = pd.DataFrame()
			output_read['RSI_apply_to'] = np.nan
			output_read['RSI_length'] = np.nan
			output_read['corr_low'] = np.nan
			output_read['corr_high'] = np.nan
			output_read['Divergence_diff_extereme'] = np.nan
			output_read['Divergence_num_exteremes_min'] = np.nan
			output_read['Divergence_num_exteremes_max'] = np.nan
			output_read['frequency'] = np.nan
			output_read['score'] = np.nan
			
		
		output = pd.DataFrame(np.ones(self.turn))
		output['RSI_apply_to'] = np.nan
		output['RSI_length'] = np.nan
		output['corr_low'] = np.nan
		output['corr_high'] = np.nan
		output['Divergence_diff_extereme'] = np.nan
		output['Divergence_num_exteremes_min'] = np.nan
		output['Divergence_num_exteremes_max'] = np.nan
		output['frequency'] = np.nan
		output['score'] = np.nan
		

		for i in range(self.turn):
			rsi_parameters.elements['RSI' + '_apply_to'] = random.choice([
																		'open',
																		'close',
																		'low',
																		'high',
																		'HL/2',
																		'HLC/3',
																		'HLCC/4',
																		'OHLC/4'
																		])
			rsi_parameters.elements['RSI' + '_length'] = randint(2, 300)

			ind_params.elements['Divergence' + '_diff_extereme'] = randint(1 , 6)
			ind_params.elements['Divergence' + '_num_exteremes_min'] = randint(2 , 500)
			ind_params.elements['Divergence' + '_num_exteremes_max'] = randint(2 , 500)

			repeat_counter = 0
			if output.dropna().empty == False:

				repeat_checker_now = np.where(
											(rsi_parameters.elements['RSI' + '_length'] == output['RSI_length'].values) &
											(ind_params.elements['Divergence' + '_diff_extereme'] == output['Divergence_diff_extereme'].values) &
											(ind_params.elements['Divergence' + '_num_exteremes_min'] == output['Divergence_num_exteremes_min'].values) &
											(ind_params.elements['Divergence' + '_num_exteremes_max'] == output['Divergence_num_exteremes_max'].values) &
											(rsi_parameters.elements['RSI' + '_apply_to'] == output['RSI_apply_to'].values)
										)[0]

				repeat_checker_before = np.where(
											(rsi_parameters.elements['RSI' + '_length'] == output_read['RSI_length'].values) &
											(ind_params.elements['Divergence' + '_diff_extereme'] == output_read['Divergence_diff_extereme'].values) &
											(ind_params.elements['Divergence' + '_num_exteremes_min'] == output_read['Divergence_num_exteremes_min'].values) &
											(ind_params.elements['Divergence' + '_num_exteremes_max'] == output_read['Divergence_num_exteremes_max'].values) &
											(rsi_parameters.elements['RSI' + '_apply_to'] == output_read['RSI_apply_to'].values)
										)[0]

				while (
						len(repeat_checker_now) > 0 or
						len(repeat_checker_before) >0
						):
					rsi_parameters.elements['RSI' + '_apply_to'] = random.choice([
																				'open',
																				'close',
																				'low',
																				'high',
																				'HL/2',
																				'HLC/3',
																				'HLCC/4',
																				'OHLC/4'
																				])
					rsi_parameters.elements['RSI' + '_length'] = randint(2, 300)

					ind_params.elements['Divergence' + '_diff_extereme'] = randint(1 , 6)
					ind_params.elements['Divergence' + '_num_exteremes_min'] = randint(2 , 250)
					ind_params.elements['Divergence' + '_num_exteremes_max'] = randint(2 , 250)

					repeat_checker_now = np.where(
											(rsi_parameters.elements['RSI' + '_length'] == output['RSI_length'].values) &
											(ind_params.elements['Divergence' + '_diff_extereme'] == output['Divergence_diff_extereme'].values) &
											(ind_params.elements['Divergence' + '_num_exteremes_min'] == output['Divergence_num_exteremes_min'].values) &
											(ind_params.elements['Divergence' + '_num_exteremes_max'] == output['Divergence_num_exteremes_max'].values) &
											(rsi_parameters.elements['RSI' + '_apply_to'] == output['RSI_apply_to'].values)
										)[0]

					repeat_checker_before = np.where(
												(rsi_parameters.elements['RSI' + '_length'] == output_read['RSI_length'].values) &
												(ind_params.elements['Divergence' + '_diff_extereme'] == output_read['Divergence_diff_extereme'].values) &
												(ind_params.elements['Divergence' + '_num_exteremes_min'] == output_read['Divergence_num_exteremes_min'].values) &
												(ind_params.elements['Divergence' + '_num_exteremes_max'] == output_read['Divergence_num_exteremes_max'].values) &
												(rsi_parameters.elements['RSI' + '_apply_to'] == output_read['RSI_apply_to'].values)
											)[0]

					if repeat_counter >= len(output_read['RSI_length'].dropna().index): break
					repeat_counter += 1
				

			output['RSI_apply_to'][i] = rsi_parameters.elements['RSI' + '_apply_to']
			output['RSI_length'][i] = rsi_parameters.elements['RSI' + '_length']
			output['frequency'][i] = freq_time

			rsi_parameters.elements['dataset_5M'] = self.dataset
			rsi_parameters.elements['dataset_1H'] = self.dataset

			rsi = RSI(parameters = rsi_parameters, config = rsi_config)
			rsi_calc = rsi.calculator_rsi()


			rsi = Divergence(parameters = ind_params, config = ind_config)
			signal, signaltype, indicator = rsi.divergence(
															sigtype = self.sigtype,
															sigpriority = self.sigpriority,
															indicator = rsi_calc,
															column_div = 'rsi',
															ind_name = 'rsi',
															dataset_5M = rsi_parameters.elements['dataset_' + self.timeframe],
															dataset_1H = rsi_parameters.elements['dataset_' + self.timeframe],
															symbol = self.symbol,
															flaglearn = False,
															flagtest = True
															)
			bar.next()

			if signal.empty == True: continue
			divergence_out = pd.DataFrame(np.ones(signal.index[-1]))
			divergence_out['rsi'] = np.nan
			divergence_out['low'] = np.nan
			divergence_out['high'] = np.nan

			counter = 0
			for elm in signal.index:
				divergence_out['rsi'][counter] = signal.indicator_front[elm]
				divergence_out['rsi'][counter + 1] = signal.indicator_back[elm]

				divergence_out['low'][counter] = signal.low_front[elm]
				divergence_out['low'][counter + 1] = signal.low_back[elm]

				divergence_out['high'][counter] = signal.high_front[elm]
				divergence_out['high'][counter + 1] = signal.high_back[elm]

				counter += 2

			divergence_out = divergence_out.dropna()
			divergence_out = divergence_out.drop(columns = [0])

			number_divergence = len(divergence_out.index)/1000

			divergence_out = divergence_out.corr()

			output['score'][i] = -((divergence_out['rsi'][2] * divergence_out['rsi'][1] * number_divergence) ** (1/3))

			if (
				divergence_out['rsi'][2] > 0 and
				divergence_out['rsi'][1] > 0
				):
				output['score'][i] = -output['score'][i]

			output['corr_low'][i] = divergence_out['rsi'][1]
			output['corr_high'][i] = divergence_out['rsi'][2]
			output['Divergence_diff_extereme'][i] = ind_params.elements['Divergence' + '_diff_extereme']
			output['Divergence_num_exteremes_min'][i] = ind_params.elements['Divergence' + '_num_exteremes_min']
			output['Divergence_num_exteremes_max'][i] = ind_params.elements['Divergence' + '_num_exteremes_max']
			#print(output.head(i))
			#print('turn = ', self.main_turn * i, ', score = ', output_read['score'].min(), ' ', self.sigtype, ' ', self.sigpriority)

		if os.path.exists(path):
			os.remove(path)

		output = output.drop(columns = [0])
		output = pd.concat([output, output_read], ignore_index=True)

		output.dropna().sort_values(by = ['score'], ascending = False).to_csv(path)

		print()
		print('RSI Optimizer ', self.sigtype, ' ', self.sigpriority, ' ', self.timeframe, ' is Finished')

		return output.dropna().sort_values(by = ['score'], ascending = False)