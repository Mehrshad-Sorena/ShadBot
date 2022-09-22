import pandas as pd
# import seaborn as sns
# from pathlib import Path
# import requests
# from io import BytesIO
# from zipfile import ZipFile, BadZipFile
# import pandas_datareader.data as web
# from talib import RSI, BBANDS, MACD, CCI, EMA, SMA, STOCH
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
import pandas_ta as ind
import numpy as np
import matplotlib.pyplot as plt

from src.indicators.MACD.Config import Config as MACDConfig
from src.indicators.MACD.Parameters import Parameters as MACDParameters
from src.indicators.MACD.MACD import MACD

from src.indicators.StochAstic.Config import Config as StochAsticConfig
from src.indicators.StochAstic.Parameters import Parameters as StochAsticParameters
from src.indicators.StochAstic.StochAstic import StochAstic

from src.indicators.RSI.Config import Config as RSIConfig
from src.indicators.RSI.Parameters import Parameters as RSIParameters
from src.indicators.RSI.RSI import RSI

from src.utils.Divergence.Divergence import Divergence
from src.utils.Divergence.Parameters import Parameters as indicator_parameters
from src.utils.Divergence.Config import Config as indicator_config

import sys
import os

if 'win' in sys.platform:
	path_slash = '\\'
elif 'linux' in sys.platform:
	path_slash = '/'



class FeatureEngineering:

	def __init__(self):

		self.symbol = 'XAUUSD_i'
		
		self.lags = [1, 2, 3, 6, 9, 12, 24, 48]
		self.timelags = [1, 2, 3, 6, 9, 12, 24, 48]
		self.momentums = [1, 2, 3, 6, 9, 12, 24, 48]


	


	

	def MainDataAdd(self, dataset, data):

		data['real'] = np.nan
		for clm in dataset.columns:
			data['real'] = dataset[clm].copy(deep = True)

		return data


#with pd.option_context('display.max_rows', None, 'display.max_columns', None):

