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

						#************** Divergence:

						'path_society': os.path.join(Path(__file__).parent , 'GeneticLearning_DB' + path_slash + 'society' + path_slash),
						'path_graveyard': os.path.join(Path(__file__).parent , 'GeneticLearning_DB' + path_slash + 'graveyard' + path_slash),
						'path_superhuman': os.path.join(Path(__file__).parent , 'GeneticLearning_DB' + path_slash + 'superhuman' + path_slash),
						'path_elites': os.path.join(Path(__file__).parent , 'GeneticLearning_DB' + path_slash + 'elites' + path_slash),

						'path_optimized_params': os.path.join(Path(__file__).parent , 'OptimizeParameters' + path_slash),

						#/////////////////////////////

						})

