from src.utils.Optimizers import Optimizers
from src.utils.DataReader.MetaTraderReader5.LoginGetData import LoginGetData as getdata
import threading
import time

class OptimizersRunner():

	def __init__(self):

		self.symbol = 'XAUUSD_i'

	def TaskMACD(self, timeframe, sigtype, sigpriority):
		loging = getdata()

		if timeframe == '5M':
			dataset, _ = loging.readall(symbol = self.symbol, number_5M = 'all', number_1H = 0)
		if timeframe == '1H':
			_, dataset = loging.readall(symbol = self.symbol, number_5M = 0, number_1H = 'all')

		optimizers = Optimizers.Optimizers() 

		optimizers.symbol = self.symbol
		optimizers.sigpriority = sigpriority
		optimizers.sigtype = sigtype
		optimizers.turn = 100
		optimizers.dataset = dataset.copy()
		optimizers.timeframe = timeframe

		job_thread = threading.Thread(target = optimizers.MacdOptimizer)

		return job_thread

	def MACDRunner(self):

		try:
		 	self.TaskMACD(timeframe = '5M', sigtype = 'buy', sigpriority = 'primary').start()
		 	self.TaskMACD(timeframe = '5M', sigtype = 'buy', sigpriority = 'secondry').start()
		 	self.TaskMACD(timeframe = '5M', sigtype = 'sell', sigpriority = 'primary').start()
		 	self.TaskMACD(timeframe = '5M', sigtype = 'sell', sigpriority = 'secondry').start()

		 	self.TaskMACD(timeframe = '1H', sigtype = 'buy', sigpriority = 'primary').start()
		 	self.TaskMACD(timeframe = '1H', sigtype = 'buy', sigpriority = 'secondry').start()
		 	self.TaskMACD(timeframe = '1H', sigtype = 'sell', sigpriority = 'primary').start()
		 	self.TaskMACD(timeframe = '1H', sigtype = 'sell', sigpriority = 'secondry').start()

		except Exception as ex:
			print('MACD Optimizer ERROR: ', ex)


	def TaskStochAstic(self, timeframe, sigtype, sigpriority):
		loging = getdata()

		if timeframe == '5M':
			dataset, _ = loging.readall(symbol = self.symbol, number_5M = 'all', number_1H = 0)
		if timeframe == '1H':
			_, dataset = loging.readall(symbol = self.symbol, number_5M = 0, number_1H = 'all')

		optimizers = Optimizers.Optimizers() 

		optimizers.symbol = self.symbol
		optimizers.sigpriority = sigpriority
		optimizers.sigtype = sigtype
		optimizers.turn = 100
		optimizers.dataset = dataset.copy()
		optimizers.timeframe = timeframe

		job_thread = threading.Thread(target = optimizers.StochAsticOptimizer)

		return job_thread

	def StochAsticRunner(self):

		try:
		 	self.TaskStochAstic(timeframe = '5M', sigtype = 'buy', sigpriority = 'primary').start()
		 	self.TaskStochAstic(timeframe = '5M', sigtype = 'buy', sigpriority = 'secondry').start()
		 	self.TaskStochAstic(timeframe = '5M', sigtype = 'sell', sigpriority = 'primary').start()
		 	self.TaskStochAstic(timeframe = '5M', sigtype = 'sell', sigpriority = 'secondry').start()

		 	self.TaskStochAstic(timeframe = '1H', sigtype = 'buy', sigpriority = 'primary').start()
		 	self.TaskStochAstic(timeframe = '1H', sigtype = 'buy', sigpriority = 'secondry').start()
		 	self.TaskStochAstic(timeframe = '1H', sigtype = 'sell', sigpriority = 'primary').start()
		 	self.TaskStochAstic(timeframe = '1H', sigtype = 'sell', sigpriority = 'secondry').start()

		except Exception as ex:
			print('StochAstic Optimizer ERROR: ', ex)


	def TaskRSI(self, timeframe, sigtype, sigpriority):
		loging = getdata()

		if timeframe == '5M':
			dataset, _ = loging.readall(symbol = self.symbol, number_5M = 'all', number_1H = 0)
		if timeframe == '1H':
			_, dataset = loging.readall(symbol = self.symbol, number_5M = 0, number_1H = 'all')

		optimizers = Optimizers.Optimizers() 

		optimizers.symbol = self.symbol
		optimizers.sigpriority = sigpriority
		optimizers.sigtype = sigtype
		optimizers.turn = 100
		optimizers.dataset = dataset.copy()
		optimizers.timeframe = timeframe

		job_thread = threading.Thread(target = optimizers.RSIOptimizer)

		return job_thread

	def RSIRunner(self):

		try:
		 	self.TaskRSI(timeframe = '5M', sigtype = 'buy', sigpriority = 'primary').start()
		 	self.TaskRSI(timeframe = '5M', sigtype = 'buy', sigpriority = 'secondry').start()
		 	self.TaskRSI(timeframe = '5M', sigtype = 'sell', sigpriority = 'primary').start()
		 	self.TaskRSI(timeframe = '5M', sigtype = 'sell', sigpriority = 'secondry').start()

		 	self.TaskRSI(timeframe = '1H', sigtype = 'buy', sigpriority = 'primary').start()
		 	self.TaskRSI(timeframe = '1H', sigtype = 'buy', sigpriority = 'secondry').start()
		 	self.TaskRSI(timeframe = '1H', sigtype = 'sell', sigpriority = 'primary').start()
		 	self.TaskRSI(timeframe = '1H', sigtype = 'sell', sigpriority = 'secondry').start()

		except Exception as ex:
			print('RSI Optimizer ERROR: ', ex)


	def Run(self):

		self.MACDRunner()
		self.StochAsticRunner()
		self.RSIRunner()

		
