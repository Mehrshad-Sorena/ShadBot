from src.utils.DataReader.MetaTraderReader5.LoginGetData import LoginGetData as getdata
from src.indicators.MACD.Parameters import Parameters
from src.indicators.MACD.Config import Config
from src.indicators.MACD.MACD import MACD
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
	parameters.elements['MACD_apply_to'] = 'close'

	optimizers = Optimizers.Optimizers() 

	optimizers.symbol = 'XAUUSD_i'
	optimizers.sigpriority = 'secondry'
	optimizers.sigtype = 'buy'
	optimizers.turn = 500
	optimizers.dataset = parameters.elements['dataset_5M']
	optimizers.timeframe = '5M'

	optimizers.MacdOptimizer()

	macd = MACD(parameters = parameters, config = config)
	macd_calc = macd.Genetic(
							dataset_5M = parameters.elements['dataset_5M'], 
							dataset_1H = parameters.elements['dataset_1H'], 
							symbol = 'XAUUSD_i',
							signaltype = 'buy', 
							signalpriority = 'secondry', 
							num_turn = 40
							)

	for turn in range(0,4):
		macd_calc = macd.GetPermit(
								dataset_5M = parameters.elements['dataset_5M'],
								dataset_1H = parameters.elements['dataset_1H'], 
								symbol = 'XAUUSD_i',
								signaltype = 'buy',
								signalpriority = 'secondry',
								flag_savepic = False
								)

		if (
			macd_calc['draw_down'][0] <= 7 &
			macd_calc['permit'][0] == True
			): 
			break