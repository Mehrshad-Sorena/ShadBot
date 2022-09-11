from src.utils.DataReader.MetaTraderReader5.LoginGetData import LoginGetData as getdata
from src.indicators.MACD.Parameters import Parameters
from src.indicators.MACD.Config import Config
from src.indicators.MACD.MACD import MACD
import pandas as pd
from src.utils.Divergence.Parameters import Parameters as IndicatorParameters
from src.utils.Divergence.Config import Config as IndicatorConfig
from src.utils.Divergence.Divergence import Divergence
from src.utils.Optimizers import Optimizers
from src.utils.Optimizers import NoiseCanceller
import os
import numpy as np
import pandas as pd
import sys

def Run():
	loging = getdata()

	parameters = Parameters()
	config = Config()

	parameters.elements['dataset_5M'], parameters.elements['dataset_1H'] = loging.readall(symbol = 'XAUUSD_i', number_5M = 'all', number_1H = 'all')
	parameters.elements['symbol'] = 'XAUUSD_i'
	parameters.elements['MACD_apply_to'] = 'close'

	dataset_5M_real, _ = loging.readall(symbol = 'XAUUSD_i', number_5M = 'all', number_1H = 0)
	dataset_5M_real = dataset_5M_real[parameters.elements['symbol']]

	optimizers = Optimizers.Optimizers() 

	noise_canceller = NoiseCanceller.NoiseCanceller()
	parameters.elements['dataset_5M']['XAUUSD_i']['close'] = noise_canceller.NoiseWavelet(dataset = parameters.elements['dataset_5M']['XAUUSD_i'], applyto = 'close')
	parameters.elements['dataset_5M']['XAUUSD_i']['open'] = noise_canceller.NoiseWavelet(dataset = parameters.elements['dataset_5M']['XAUUSD_i'], applyto = 'open')
	parameters.elements['dataset_5M']['XAUUSD_i']['high'] = noise_canceller.NoiseWavelet(dataset = parameters.elements['dataset_5M']['XAUUSD_i'], applyto = 'high')
	parameters.elements['dataset_5M']['XAUUSD_i']['low'] = noise_canceller.NoiseWavelet(dataset = parameters.elements['dataset_5M']['XAUUSD_i'], applyto = 'low')
	parameters.elements['dataset_5M']['XAUUSD_i']['HL/2'] = noise_canceller.NoiseWavelet(dataset = parameters.elements['dataset_5M']['XAUUSD_i'], applyto = 'HL/2')
	parameters.elements['dataset_5M']['XAUUSD_i']['HLC/3'] = noise_canceller.NoiseWavelet(dataset = parameters.elements['dataset_5M']['XAUUSD_i'], applyto = 'HLC/3')
	parameters.elements['dataset_5M']['XAUUSD_i']['HLCC/4'] = noise_canceller.NoiseWavelet(dataset = parameters.elements['dataset_5M']['XAUUSD_i'], applyto = 'HLCC/4')
	parameters.elements['dataset_5M']['XAUUSD_i']['OHLC/4'] = noise_canceller.NoiseWavelet(dataset = parameters.elements['dataset_5M']['XAUUSD_i'], applyto = 'OHLC/4')

	parameters.elements['dataset_1H']['XAUUSD_i']['close'] = noise_canceller.NoiseWavelet(dataset = parameters.elements['dataset_1H']['XAUUSD_i'], applyto = 'close')
	parameters.elements['dataset_1H']['XAUUSD_i']['open'] = noise_canceller.NoiseWavelet(dataset = parameters.elements['dataset_1H']['XAUUSD_i'], applyto = 'open')
	parameters.elements['dataset_1H']['XAUUSD_i']['high'] = noise_canceller.NoiseWavelet(dataset = parameters.elements['dataset_1H']['XAUUSD_i'], applyto = 'high')
	parameters.elements['dataset_1H']['XAUUSD_i']['low'] = noise_canceller.NoiseWavelet(dataset = parameters.elements['dataset_1H']['XAUUSD_i'], applyto = 'low')
	parameters.elements['dataset_1H']['XAUUSD_i']['HL/2'] = noise_canceller.NoiseWavelet(dataset = parameters.elements['dataset_1H']['XAUUSD_i'], applyto = 'HL/2')
	parameters.elements['dataset_1H']['XAUUSD_i']['HLC/3'] = noise_canceller.NoiseWavelet(dataset = parameters.elements['dataset_1H']['XAUUSD_i'], applyto = 'HLC/3')
	parameters.elements['dataset_1H']['XAUUSD_i']['HLCC/4'] = noise_canceller.NoiseWavelet(dataset = parameters.elements['dataset_1H']['XAUUSD_i'], applyto = 'HLCC/4')
	parameters.elements['dataset_1H']['XAUUSD_i']['OHLC/4'] = noise_canceller.NoiseWavelet(dataset = parameters.elements['dataset_1H']['XAUUSD_i'], applyto = 'OHLC/4')

	print('Data Ready .... ')
	# sys.exit()

	optimizers.symbol = 'XAUUSD_i'
	optimizers.sigpriority = 'primary'
	optimizers.sigtype = 'buy'
	optimizers.turn = 100
	optimizers.dataset = parameters.elements['dataset_5M'].copy()
	optimizers.timeframe = '5M'

	optimizers.MacdOptimizer()

	#parameters.elements['dataset_5M'], parameters.elements['dataset_1H'] = loging.readall(symbol = 'XAUUSD_i', number_5M = 'all', number_1H = 'all')

	macd = MACD(parameters = parameters, config = config)

	try:
		macd_calc = macd.Genetic(
								dataset_5M_real = dataset_5M_real,
								dataset_5M = parameters.elements['dataset_5M'], 
								dataset_1H = parameters.elements['dataset_1H'], 
								symbol = 'XAUUSD_i',
								signaltype = 'buy', 
								signalpriority = 'primary', 
								num_turn = 40
								)

	except Exception as ex:
		print('MACD ERROR: ', ex)

	for turn in range(0,4):
		print('GetPermit ... ')
		try:
			macd_calc = macd.GetPermit(
									dataset_5M_real = dataset_5M_real,
									dataset_5M = parameters.elements['dataset_5M'],
									dataset_1H = parameters.elements['dataset_1H'], 
									symbol = 'XAUUSD_i',
									signaltype = 'buy',
									signalpriority = 'primary',
									flag_savepic = False
									)
			print()

			if (
				macd_calc['draw_down'][0] <= 7 &
				macd_calc['permit'][0] == True
				): 
				break

		except Exception as ex:
			print('MACD GetPermit ERROR: ', ex)