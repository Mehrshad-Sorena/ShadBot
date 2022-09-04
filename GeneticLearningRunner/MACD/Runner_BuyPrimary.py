from src.utils.DataReader.MetaTraderReader5.LoginGetData import LoginGetData as getdata
from src.indicators.MACD.Parameters import Parameters
from src.indicators.MACD.Config import Config
from src.indicators.MACD.MACD import MACD
import pandas as pd
from src.utils.Divergence.Parameters import Parameters as IndicatorParameters
from src.utils.Divergence.Config import Config as IndicatorConfig
from src.utils.Divergence.Divergence import Divergence
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

	macd = MACD(parameters = parameters, config = config)
	macd_calc = macd.Genetic(
							dataset_5M = parameters.elements['dataset_5M'], 
							dataset_1H = parameters.elements['dataset_1H'], 
							symbol = 'XAUUSD_i',
							signaltype = 'buy', 
							signalpriority = 'primary', 
							num_turn = 40
							)

	macd_calc = macd.GetPermit(
							dataset_5M = parameters.elements['dataset_5M'],
							dataset_1H = parameters.elements['dataset_1H'], 
							symbol = 'XAUUSD_i',
							signaltype = 'buy',
							signalpriority = 'primary',
							flag_savepic = False
							)