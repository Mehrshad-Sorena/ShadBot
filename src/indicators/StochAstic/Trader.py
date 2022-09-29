from src.utils.DataReader.MetaTraderReader5.LoginGetData import LoginGetData as getdata
from src.utils.Tools.carrier import carrier_buy, carrier_sell
from .BascketManager import basket_manager_stochastic_div
from datetime import datetime
from src.utils.ForexNews.forex_news import news

try:
	import MetaTrader5 as mt5
except Exception as ex:
	print(ex)
	
import pandas as pd
import numpy as np
import json
import time
import os

from src.indicators.StochAstic.Parameters import Parameters as StochAsticParameters
from src.indicators.StochAstic.Config import Config as StochAsticConfig
from src.indicators.StochAstic.StochAstic import StochAstic




symbol_black_list = np.array(
	[
		'WSt30_m_i','SPX500_m_i','NQ100_m_i','GER40_m_i',
		'GER40_i','USDRUR','USDRUR_i','USDRUB','USDRUB_i',
		'USDHKD','WTI_i','BRN_i','STOXX50_i','NQ100_i',
		'NG_i','HSI50_i','CAC40_i','ASX200_i','SPX500_i',
		'NIKK225_i','IBEX35_i','FTSE100_i','RUBRUR',
		'EURDKK_i','DAX30_i','XRPUSD_i','XBNUSD_i',
		'LTCUSD_i','ETHUSD_i','BTCUSD_i','_DXY','_DJI',
		'EURTRY_i','USDTRY_i','USDDKK_i'
	])

def get_all_deta_online(symbol, account_name):

	loging = getdata()
	loging.account_name = account_name
	loging.initilizer()
	loging.login()

	symbol_data_5M = loging.getone(timeframe = '5M', number = 1200, symbol = symbol)
	symbol_data_1H = loging.getone(timeframe = '1H', number = 1005, symbol = symbol)

	symbol = loging.get_symbols()

	money = loging.get_balance()

	return symbol_data_5M, symbol_data_1H, symbol, money


def trader_stochastic_div(
					symbol_data_5M,
					symbol_data_1H,
					symbol,
					money,
					account_name
					):
	forexnews_path = 'forexnews.json'
	time_last = time.time()

	for sym in symbol:

		#if np.where(sym.name == symbol_black_list)[0].size != 0: continue

		if not (
			# sym.name == 'AUDCAD_i' or
			# sym.name == 'AUDCHF_i' or
			# sym.name == 'AUDUSD_i' or
			# sym.name == 'CADJPY_i' or
			# sym.name == 'EURAUD_i' or
			# sym.name == 'EURCAD_i' or
			# sym.name == 'EURCHF_i' or
			# sym.name == 'EURGBP_i' or
			# sym.name == 'EURUSD_i' or
			# sym.name == 'EURJPY_i' or
			# sym.name == 'GBPAUD_i' or
			# sym.name == 'GBPCAD_i' or
			# sym.name == 'GBPJPY_i' or
			# sym.name == 'GBPUSD_i' or
			# sym.name == 'USDJPY_i' or
			# sym.name == 'USDCAD_i' or
			# sym.name == 'CAC40_i' or
			# sym.name == 'FTSE100_i' or
			# sym.name == 'GER40_i' or
			# sym.name == 'WSt30_m_i' or
			# sym.name == 'STOXX50_i' or
			# sym.name == 'CHNA50_m_i' or
			# sym.name == 'HSI50_i' or
			# sym.name == 'NQ100_i' or
			# sym.name == 'LTCUSD_i' or
			# sym.name == 'XRPUSD_i' or
			# sym.name == 'BTCUSD_i' or
			#sym.name == 'ETHUSD_i'
			sym.name == 'XAUUSD_i'
			): continue

		if os.path.exists(forexnews_path):
			with open(forexnews_path, 'r') as file:
				forex_news = json.loads(file.read())

			now = datetime.now()
			for fn in forex_news.keys():
				if fn in sym.name:
					hour = forex_news.get(fn).get('hour')
					minute = forex_news.get(fn).get('min')
					impact = forex_news.get(fn).get('impact')
				else:
					impact = None
			
			if impact == 'medium' or impact == 'high':
				time_now_min = now.hour*60 + now.minute
				time_forexnews_min = hour*60 + minute
				if time_forexnews_min-30 < time_now_min < time_forexnews_min+30: continue
		else:
			news()


		stochastic_parameters = StochAsticParameters()
		stochastic_config = StochAsticConfig()

		stochastic_parameters.elements['dataset_5M'] = symbol_data_5M
		stochastic_parameters.elements['dataset_1H'] = symbol_data_1H
		stochastic_parameters.elements['symbol'] = sym.name

		stochastic = StochAstic(parameters = stochastic_parameters, config = stochastic_config)
		signal, tp, st = stochastic.LastSignal(
										dataset_5M = stochastic_parameters.elements['dataset_5M'], 
										dataset_1H = stochastic_parameters.elements['dataset_1H'], 
										symbol = sym.name,
										)

		lot = basket_manager_stochastic_div(symbols=symbol,symbol=sym.name,my_money=money,signal=signal, account_name = account_name)

		if lot > 0.09: lot = 0.09
		if 0 < lot < 0.01: lot = 0.01

		if (
			sym.name == 'CAC40_i' or
			sym.name == 'CHNA50_m_i' or
			sym.name == 'FTSE100_i' or
			sym.name == 'GER40_i' or
			sym.name == 'HSI50_i' or
			sym.name == 'NQ100_i' or
			sym.name == 'STOXX50_i' or
			sym.name == 'WSt30_m_i'
			):
			
			lot = lot * 0#10 
			lot = float("{:.1f}".format((lot)))

		print('================> ',sym.name)
		print('signal =  ',signal)
		print('tp: ',tp)
		print('st: ',st)
		print('lot: ',lot)
		print('================================')

		if lot:
			if (
				signal == 'buy_primary' or
				signal == 'buy_secondry'
				):
				carrier_buy(symbol=sym.name,lot=lot,st=st,tp=tp,comment='stoch div'+signal,magic=time.time_ns())
			elif (
				signal == 'sell_primary' or
				signal == 'sell_secondry'
				):
				carrier_sell(symbol=sym.name,lot=lot,st=st,tp=tp,comment='stoch div'+signal,magic=time.time_ns())
			elif signal == 'no_trade':
				continue
		else:
			continue
	return

def trader_task_stochastic_div(symbol, account_name):
	try:
		print('****************** Start StochAstic *************************')
		symbol_data_5M,symbol_data_1H,symbol,money = get_all_deta_online(symbol = symbol, account_name = account_name)
		trader_stochastic_div(
						symbol_data_5M = symbol_data_5M,
						symbol_data_1H = symbol_data_1H,
						symbol = symbol,
						money = money,
						account_name = account_name
						)
		print('****************** Finish StochAstic *************************')
	except Exception as ex:
		print('===== Trader StochAstic Error ===> ',ex)
	return