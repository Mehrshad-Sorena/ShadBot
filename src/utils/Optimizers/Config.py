from pathlib import Path, PurePosixPath
import os
import sys


if 'win' in sys.platform:
	path_slash = '\\'
elif 'linux' in sys.platform:
	path_slash = '/'

class Config:

	def __init__(cls):
		
		cls.cfg = dict(
						{
						#Config For ExtremePoints:
						'path_MACD': os.path.join(Path(__file__).parent.parent.parent , 'indicators' + path_slash + 'MACD' + path_slash +'OptimizeParameters'),
						'path_StochAstic': os.path.join(Path(__file__).parent.parent.parent , 'indicators' + path_slash + 'StochAstic' + path_slash +'OptimizeParameters'),
						'path_RSI': os.path.join(Path(__file__).parent.parent.parent , 'indicators' + path_slash + 'RSI' + path_slash +'OptimizeParameters'),
						#///////////////////////////
						}
						)