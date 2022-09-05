from src.utils.DataReader.MetaTraderReader5.LoginGetData import LoginGetData as getdata
from src.utils.Optimizers import OptimizersRunner

loging = getdata()

dataset_5M, dataset_1H = loging.readall(symbol = 'XAUUSD_i', number_5M = 'all', number_1H = 'all')

optimizersrunner = OptimizersRunner.OptimizersRunner() 

optimizersrunner.symbol = 'XAUUSD_i'

optimizersrunner.Run()