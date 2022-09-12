from .Config import Config
import pandas as pd
import os


#Functions:

#Read()
#Write()
#Delete()

#*************

class DatasetIO:

	def Read(self, name, symbol):

		dataset_config = Config()

		path = dataset_config.cfg['path_' + name] + symbol + '.csv'

		if os.path.exists(path):
			dataset = pd.read_csv(path)

			if 'Unnamed' in dataset.columns[0]:
				dataset = dataset.drop(columns = ['Unnamed: 0'])

		else:
			dataset = pd.DataFrame()

		return dataset


	def Write(self, dataset, name, symbol):

		dataset_config = Config()

		path = dataset_config.cfg['path_' + name] + symbol + '.csv'

		if os.path.exists(dataset_config.cfg['path_' + name]):

			if os.path.exists(path):
				os.remove(path)
			dataset = dataset.to_csv(path)

		else:
			os.makedirs(dataset_config.cfg['path_' + name])
			dataset = dataset.to_csv(path)

	def Delete(self, name, symbol):

		dataset_config = Config()

		path = dataset_config.cfg['path_' + name] + symbol + '.csv'

		if os.path.exists(path):
			os.remove(path)