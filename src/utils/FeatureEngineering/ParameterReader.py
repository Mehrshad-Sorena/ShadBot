from src.indicators.StochAstic.Parameters import Parameters as StochAsticParameters
from src.utils.Divergence.Parameters import Parameters as indicator_parameters
from src.indicators.StochAstic.Config import Config as StochAsticConfig
from src.indicators.MACD.Parameters import Parameters as MACDParameters
from src.indicators.RSI.Parameters import Parameters as RSIParameters
from src.utils.Divergence.Config import Config as indicator_config
from src.indicators.MACD.Config import Config as MACDConfig
from src.indicators.RSI.Config import Config as RSIConfig
import pandas as pd
import sys
import os

if 'win' in sys.platform:
	path_slash = '\\'
elif 'linux' in sys.platform:
	path_slash = '/'


class ParameterReader():

	#Parameter Reader:
	def Divergence(
					self,
					signalpriority,
					signaltype,
					symbol,
					timeframe = '5M',
					dataset = pd.DataFrame(),
					indicator_name = 'macd'
					):

		if indicator_name == 'macd':

			macd_config = MACDConfig()
			macd_parameters = MACDParameters()
			ind_config = indicator_config()
			ind_parameters = indicator_parameters()

			if os.path.exists(
								macd_config.cfg['path_optimized_params'] + 
								signalpriority + path_slash + 
								signaltype + path_slash + 
								timeframe + path_slash + 
								symbol + '.csv'
							):

				macd_optimizer_params = pd.read_csv(
													macd_config.cfg['path_optimized_params'] + 
													signalpriority + path_slash + 
													signaltype + path_slash + 
													timeframe + path_slash + 
													symbol + '.csv'
													)

				ind_parameters.elements['Divergence_num_exteremes_min'] = int(macd_optimizer_params['Divergence_num_exteremes_min'].iloc[-1])
				ind_parameters.elements['Divergence_num_exteremes_max'] = int(macd_optimizer_params['Divergence_num_exteremes_max'].iloc[-1])
				ind_parameters.elements['Divergence_diff_extereme'] = int(macd_optimizer_params['Divergence_diff_extereme'].iloc[-1])

				ind_parameters.elements['dataset_5M'] = dataset
				ind_parameters.elements['dataset_1H'] = dataset

				macd_parameters.elements['MACD_apply_to'] = macd_optimizer_params['MACD_apply_to'].iloc[-1]
				macd_parameters.elements['MACD_fast'] = int(macd_optimizer_params['MACD_fast'].iloc[-1])
				macd_parameters.elements['MACD_slow'] = int(macd_optimizer_params['MACD_slow'].iloc[-1])
				macd_parameters.elements['MACD_signal'] = int(macd_optimizer_params['MACD_signal'].iloc[-1])
				macd_parameters.elements['MACD_column_div'] = macd_optimizer_params['MACD_column_div'].iloc[-1]

				macd_parameters.elements['dataset_5M'] = dataset
				macd_parameters.elements['dataset_1H'] = dataset
				macd_parameters.elements['symbol'] = symbol

			else:
				return False

			return macd_parameters, macd_config, ind_parameters, ind_config

		elif indicator_name == 'stochastic':

			stochastic_config = StochAsticConfig()
			stochastic_parameters = StochAsticParameters()
			ind_config = indicator_config()
			ind_parameters = indicator_parameters()

			if os.path.exists(
								stochastic_config.cfg['path_optimized_params'] + 
								signalpriority + path_slash + 
								signaltype + path_slash + 
								timeframe + path_slash + 
								symbol + '.csv'
							):

				stochastic_optimizer_params = pd.read_csv(
														stochastic_config.cfg['path_optimized_params'] + 
														signalpriority + path_slash + 
														signaltype + path_slash + 
														timeframe + path_slash + 
														symbol + '.csv'
														)

				ind_parameters.elements['Divergence_num_exteremes_min'] = int(stochastic_optimizer_params['Divergence_num_exteremes_min'].iloc[-1])
				ind_parameters.elements['Divergence_num_exteremes_max'] = int(stochastic_optimizer_params['Divergence_num_exteremes_max'].iloc[-1])
				ind_parameters.elements['Divergence_diff_extereme'] = int(stochastic_optimizer_params['Divergence_diff_extereme'].iloc[-1])

				ind_parameters.elements['dataset_5M'] = dataset
				ind_parameters.elements['dataset_1H'] = dataset

				stochastic_parameters.elements['StochAstic_d'] = int(stochastic_optimizer_params['StochAstic_d'].iloc[-1])
				stochastic_parameters.elements['StochAstic_k'] = int(stochastic_optimizer_params['StochAstic_k'].iloc[-1])
				stochastic_parameters.elements['StochAstic_smooth_k'] = int(stochastic_optimizer_params['StochAstic_smooth_k'].iloc[-1])
				stochastic_parameters.elements['StochAstic_column_div'] = stochastic_optimizer_params['StochAstic_column_div'].iloc[-1]
				stochastic_parameters.elements['StochAstic_mamod'] = stochastic_optimizer_params['StochAstic_mamod'].iloc[-1]

				stochastic_parameters.elements['dataset_5M'] = dataset
				stochastic_parameters.elements['dataset_1H'] = dataset
				stochastic_parameters.elements['symbol'] = symbol
			else:
				return False

			return stochastic_parameters, stochastic_config, ind_parameters, ind_config

		elif indicator_name == 'rsi':

			rsi_config = RSIConfig()
			rsi_parameters = RSIParameters()
			ind_config = indicator_config()
			ind_parameters = indicator_parameters()

			if os.path.exists(
								rsi_config.cfg['path_optimized_params'] + 
								signalpriority + path_slash + 
								signaltype + path_slash + 
								timeframe + path_slash + 
								symbol + '.csv'
							):

				rsi_optimizer_params = pd.read_csv(
													rsi_config.cfg['path_optimized_params'] + 
													signalpriority + path_slash + 
													signaltype + path_slash + 
													timeframe + path_slash + 
													symbol + '.csv'
													)

				ind_parameters.elements['Divergence_num_exteremes_min'] = int(rsi_optimizer_params['Divergence_num_exteremes_min'].iloc[-1])
				ind_parameters.elements['Divergence_num_exteremes_max'] = int(rsi_optimizer_params['Divergence_num_exteremes_max'].iloc[-1])
				ind_parameters.elements['Divergence_diff_extereme'] = int(rsi_optimizer_params['Divergence_diff_extereme'].iloc[-1])

				ind_parameters.elements['dataset_5M'] = dataset
				ind_parameters.elements['dataset_1H'] = dataset

				rsi_parameters.elements['RSI_apply_to'] = rsi_optimizer_params['RSI_apply_to'].iloc[-1]
				rsi_parameters.elements['RSI_length'] = int(rsi_optimizer_params['RSI_length'].iloc[-1])

				rsi_parameters.elements['dataset_5M'] = dataset
				rsi_parameters.elements['dataset_1H'] = dataset
				rsi_parameters.elements['symbol'] = symbol

			else:
				return False

			return rsi_parameters, rsi_config, ind_parameters, ind_config

	#////////////////////////////////