from pathlib import Path, PurePosixPath
import os
import sys


if 'win' in sys.platform:
	path_slash = '\\'
elif 'linux' in sys.platform:
	path_slash = '/'



class Config:

	def __init__(cls):
		
		cls.cfg = dict({

						#************** Feature Engineering:

						'path_frequency': os.path.join(Path(__file__).parent , 'Dataset' + path_slash + 'Frequencies' + path_slash),
						'path_fourier': os.path.join(Path(__file__).parent , 'Dataset' + path_slash + 'Fouriers' + path_slash),
						'path_pattern': os.path.join(Path(__file__).parent , 'Dataset' + path_slash + 'Patterns' + path_slash),
						'path_main': os.path.join(Path(__file__).parent , 'Dataset' + path_slash + 'Main' + path_slash),
						'path_lags': os.path.join(Path(__file__).parent , 'Dataset' + path_slash + 'Lag' + path_slash),

						#/////////////////////////////

						'show_bar': False,

						})