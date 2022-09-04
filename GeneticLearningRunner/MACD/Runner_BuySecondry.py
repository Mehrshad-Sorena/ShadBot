from src.Utils.DataReader.MetaTraderReader5.LoginGetData import LoginGetData as getdata
from src.Indicators.MACD.Parameters import Parameters
from src.Indicators.MACD.Config import Config
from src.Indicators.MACD.MACD import MACD
import pandas as pd
from src.Utils.Divergence.Parameters import Parameters as IndicatorParameters
from src.Utils.Divergence.Config import Config as IndicatorConfig
from src.Utils.Divergence.Divergence import Divergence
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
							signalpriority = 'secondry', 
							num_turn = 40
							)

	macd_calc = macd.GetPermit(
							dataset_5M = parameters.elements['dataset_5M'],
							dataset_1H = parameters.elements['dataset_1H'], 
							symbol = 'XAUUSD_i',
							signaltype = 'buy',
							signalpriority = 'secondry',
							flag_savepic = False
							)