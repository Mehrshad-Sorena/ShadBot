from pathlib import Path, PurePosixPath

class Config:

	def __init__(cls):
		
		cls.cfg = dict({

						#************** Divergence:

						'path_society': str(PurePosixPath(str(PurePosixPath(__file__).parent) + '/GeneticLearning_DB/society/')),
						'path_graveyard': str(PurePosixPath(str(Path(__file__).parent) + '/GeneticLearning_DB/graveyard/')),
						'path_superhuman': str(PurePosixPath(str(Path(__file__).parent) + '/GeneticLearning_DB/superhuman/')),
						'path_elites': str(PurePosixPath(str(Path(__file__).parent) + '/GeneticLearning_DB/elites/')),

						#/////////////////////////////

						})

config = Config()

print(config.cfg['path_society'] + '/XAUUSD_i' + '.csv')

import pandas as pd

print(pd.read_csv(config.cfg['path_society'] + '/primary/buy/XAUUSD_i' + '.csv'))

