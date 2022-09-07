from src.utils.DataReader.MetaTraderReader5.LoginGetData import LoginGetData as getdata
from src.indicators.RSI.Parameters import Parameters
from src.indicators.RSI.Config import Config
from src.indicators.RSI.RSI import RSI
import pandas as pd
from src.utils.Divergence.Parameters import Parameters as IndicatorParameters
from src.utils.Divergence.Config import Config as IndicatorConfig
from src.utils.Divergence.Divergence import Divergence
from src.utils.Optimizers import Optimizers
import os
import numpy as np
import pandas as pd

def Run():
	loging = getdata()


	parameters = Parameters()
	config = Config()

	parameters.elements['dataset_5M'], parameters.elements['dataset_1H'] = loging.readall(symbol = 'XAUUSD_i', number_5M = 'all', number_1H = 'all')
	parameters.elements['symbol'] = 'XAUUSD_i'
	parameters.elements['RSI_apply_to'] = 'close'

	optimizers = Optimizers.Optimizers() 

	optimizers.symbol = 'XAUUSD_i'
	optimizers.sigpriority = 'primary'
	optimizers.sigtype = 'sell'
	optimizers.turn = 100
	optimizers.dataset = parameters.elements['dataset_5M'].copy()
	optimizers.timeframe = '5M'

	optimizers.RSIOptimizer()

	rsi = RSI(parameters = parameters, config = config)

	try:
		rsi_calc = rsi.Genetic(
								dataset_5M = parameters.elements['dataset_5M'], 
								dataset_1H = parameters.elements['dataset_1H'], 
								symbol = 'XAUUSD_i',
								signaltype = 'sell', 
								signalpriority = 'primary', 
								num_turn = 40
								)

	except Exception as ex:
		print('RSI ERROR: ', ex)

	for turn in range(0,4):

		try:
			rsi_calc = rsi.GetPermit(
									dataset_5M = parameters.elements['dataset_5M'],
									dataset_1H = parameters.elements['dataset_1H'], 
									symbol = 'XAUUSD_i',
									signaltype = 'sell',
									signalpriority = 'primary',
									flag_savepic = False
									)

			if (
				rsi_calc['draw_down'][0] <= 7 &
				rsi_calc['permit'][0] == True
				): 
				break

		except Exception as ex:
			print('RSI GetPermit ERROR: ', ex)