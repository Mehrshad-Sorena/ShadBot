from src.utils.DataReader.MetaTraderReader5.LoginGetData import LoginGetData as getdata
from .Parameters import Parameters
from .Config import Config
from .RSI import RSI
import pandas as pd
import os


loging = getdata()


parameters = Parameters()
config = Config()

parameters.elements['dataset_5M'], parameters.elements['dataset_1H'] = loging.readall(symbol = 'ETHUSD_i', number_5M = 'all', number_1H = 'all')
parameters.elements['symbol'] = 'ETHUSD_i'
parameters.elements['RSI_apply_to'] = 'close'

rsi = RSI(parameters = parameters, config = config)
rsi_signal = rsi.LastSignal(
							dataset_5M = parameters.elements['dataset_5M'], 
							dataset_1H = parameters.elements['dataset_1H'], 
							symbol = 'ETHUSD_i',
							)

print(rsi_signal)
