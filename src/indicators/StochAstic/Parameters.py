import pandas as pd

class Parameters:

	def __new__(cls, *args, **kwargs):
		obj = super().__new__(cls, *args, **kwargs)
		return obj

	def __init__(self):

		self.elements = dict(
							{
							
							#*********** Divergence:

							'StochAstic_k': 3,

							'StochAstic_d': 14,
							'StochAstic_smooth_k': 5,
							'StochAstic_mamod': 'sma',

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