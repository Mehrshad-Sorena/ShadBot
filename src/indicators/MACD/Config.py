from pathlib import Path
print(type(str(Path(__file__).parent / 'GeneticLearning_DB/elites/')))
class Config:

	def __init__(cls):
		
		cls.cfg = dict({

						#************** Divergence:

						'path_society': str(Path(__file__).parent / 'GeneticLearning_DB/society/'),
						'path_graveyard': str(Path(__file__).parent / 'GeneticLearning_DB/graveyard/'),
						'path_superhuman': str(Path(__file__).parent / 'GeneticLearning_DB/superhuman/'),
						'path_elites': str(Path(__file__).parent / 'GeneticLearning_DB/elites/'),

						#/////////////////////////////

						})
