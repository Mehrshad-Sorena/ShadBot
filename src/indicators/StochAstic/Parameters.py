import pandas as pd

class Parameters:

	def __new__(cls, *args, **kwargs):
		obj = super().__new__(cls, *args, **kwargs)
		return obj

	def __init__(self):

		self.elements = dict(
							{
							
							#*********** StochAstic:

							'StochAstic' + '_k': 5,
							'StochAstic' + '_d': 3,
							'StochAstic' + '_smooth_k': 3,
							'StochAstic' + '_mamod': 'sma',

							#///////////////////////


							#ST TP Limits:

							'st_percent_up': 100,
							'st_percent_down': 80,
							'tp_percent_up': 100,
							'tp_percent_down': 80,

							#////////////////////


							#*********** Global:

							'symbol': 'XAUUSD_i',

							#//////////////////
							})