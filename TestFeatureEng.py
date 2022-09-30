from src.utils.DataReader.MetaTraderReader5.LoginGetData import LoginGetData as getdata
from src.utils.FeatureEngineering.FeatureEngineering import FeatureEngineering
from src.utils.Optimizers.NoiseCanceller import NoiseCanceller
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import sys
import seaborn as sns
import warnings as warnings
warnings.filterwarnings("ignore")

loging = getdata()

dataset_5M = pd.DataFrame()
dataset_1H = pd.DataFrame()
dataset = pd.DataFrame()

# dataset_5M, dataset_1H = loging.readall(symbol = 'XAUUSD_i', number_5M = 'all', number_1H = 0)

from src.utils.FeatureEngineering.MainFeatures import MainFeatures

main_features = MainFeatures()
main_features.symbol = 'XAUUSD_i'
print('dataset geted')

dataset_5M = main_features.Get(
							symbol = main_features.symbol,
							dataset_5M = dataset_5M, 
							dataset_1H = dataset_1H,
							mode = None
							)

# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
# 	print(dataset_5M)

# dataset_5M = dataset_5M['XAUUSD_i']
dataset_5M.index = pd.to_datetime(dataset_5M['time_5m'])

#Spliting Without Return:

dataset = dataset_5M.drop(columns = ['time_5m', 'time_1h']).copy(deep = True)
dataset = dataset.copy(deep = True).resample(str(20) + 'T').last().dropna(subset=['close_5m'])
dataset = dataset.pct_change(2).dropna()#subset=['close_5m'])
dataset = (dataset.drop(columns = ['volume_5m', 'volume_1h']) * 100).assign(volume_5m = dataset['volume_5m'], volume_1h = dataset['volume_1h'])

print(dataset)

# close = dataset['close']
# bad_close = close == -9999.0
# close[bad_close] = 0.0

column_indices = {name: i for i, name in enumerate(dataset.columns)}

n = len(dataset)

train_df = dataset[0:int(n*0.7)]
val_df = dataset[int(n*0.7):int(n*0.9)]
test_df = dataset[int(n*0.9):]

num_features = dataset.shape[1]


train_mean = train_df.mean()
train_std = train_df.std()

# train_df = (train_df - train_mean) / train_std
# val_df = (val_df - train_mean) / train_std
# test_df = (test_df - train_mean) / train_std


# df_std = (dataset - train_mean) / train_std
# df_std = df_std.melt(var_name='Column', value_name='Normalized')
# plt.figure(figsize=(12, 6))
# ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
# _ = ax.set_xticklabels(dataset.keys(), rotation=90)

# plt.show()



# sys.exit(train_df)

#/////////////////////////////////////////////////

print('data geted....')

# dataset = pd.DataFrame()

# dataset['return_1'] = dataset_5M.drop(columns = ['time', 'XAUUSD_i']).pct_change().stack()
# dataset['return_2'] = dataset_5M.drop(columns = ['time', 'XAUUSD_i']).pct_change(2).stack()
# dataset_return = dataset.swaplevel()

# print('data pct changed....')


#Spliting Data: *********************************************************************************

# column_indices = {name: i for i, name in enumerate(dataset_return.columns)}

# print('column_indices geted .....')

# index_indices = list(pd.DataFrame(name[0] for i, name in enumerate(dataset_return.index)).drop_duplicates(keep = 'first')[0])

# print('index_indices geted....: ', len(index_indices))

# time_incices = dataset_return.index.get_level_values(1)

# n = len(dataset_return.loc[index_indices[0]]) * len(index_indices)

# idx = pd.IndexSlice

# train_df = dataset_return.loc[idx[:, time_incices[0:int(n*0.7)]], dataset_return.columns]

# print('train_df maked ....: ', len(train_df.index))
# #with pd.option_context('display.max_rows', None, 'display.max_columns', None):

# val_df = dataset_return.loc[idx[:, time_incices[int(n*0.7):int(n*0.9)]], dataset_return.columns]

# print('val_df maked ....: ', len(val_df.index))

# test_df = dataset_return.loc[idx[:, time_incices[int(n*0.9):]], dataset_return.columns]

# print('test_df maked ....: ', len(test_df.index))
# num_features = dataset_return.shape[1]

#//////////////////////////////////////////////


#Createing Windoses: ******************************************************************

from src.utils.FeatureEngineering.WindowGenerator import WindowGenerator

# w1 = WindowGenerator(
# 					input_width=24, 
# 					label_width=1, 
# 					shift=24,
# 					train_df = train_df,
# 					val_df = val_df,
# 					test_df = test_df,
#                     label_columns=['return_2']
#                     )

# w2 = WindowGenerator(
# 					input_width = 6, 
# 					label_width = 1, 
# 					shift = 1,
# 					train_df = train_df,
# 					val_df = val_df,
# 					test_df = test_df,
#                     label_columns=['return_1' , 'return_2']
#                     )

# single_step_window = WindowGenerator(
# 									input_width = 1, 
# 									label_width = 1, 
# 									shift = 1,
# 									train_df = train_df,
# 									val_df = val_df,
# 									test_df = test_df,
# 				                    label_columns=['return_1' , 'return_2']
# 				                    )

wide_window = WindowGenerator(
								input_width = 240 + 23, 
								label_width = 240, 
								shift = 1,
								train_df = train_df,
								val_df = val_df,
								test_df = test_df,
			                    #label_columns=['close']
			                    )

#////////////////////////////////////////////////////////////////////////////////

# print(w1)

# Stack three slices, the length of the total window.

#print('w2 Total Size = ', w2.total_window_size * len(index_indices))

# time_incices = train_df.index.get_level_values(1)
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
# 	print(train_df)

# print(train_df[ : w2.total_window_size])
# print(train_df[100 : 100 + w2.total_window_size])
# print(train_df[200 : 200 + w2.total_window_size])

# sys.exit()
# example_window = tf.stack([
# 							np.array(train_df.loc[idx[ 'close' , time_incices[ : w2.total_window_size * len(index_indices)]], :]),
#                            	np.array(train_df.loc[idx[ 'close' , time_incices[100 : 98 + (w2.total_window_size * len(index_indices))]], :]),
#                            	np.array(train_df.loc[idx[ 'close' , time_incices[200 : 198 + (w2.total_window_size * len(index_indices))]], :])
#                            	])

# print(example_window)

# example_inputs, example_labels = w2.split_window(example_window)
# w2.example = example_inputs, example_labels

# print(example_inputs)
# print(example_labels)

#Plot Window: **********************************************************

# w1.PriceName = 'high'

# w1.plot(plot_col = 'return_2')

# print(w2.train.take(1))

#////////////////////////////////////////

#BaseLine Model: *************************************************************

from src.utils.FeatureEngineering.BaseLine import BaseLine

# print()
# print('BaseLine Model Started .............')
# print()

# baseline = BaseLine(label_index = column_indices['close'])

# baseline.compile(
# 				loss=tf.keras.losses.MeanSquaredError(),
# 				metrics=[tf.keras.metrics.MeanAbsoluteError()]
# 				)

# val_performance = {}
# performance = {}
# val_performance['Baseline'] = baseline.evaluate(wide_window.val)
# performance['Baseline'] = baseline.evaluate(wide_window.test, verbose=0)

# print('Input shape:', wide_window.example[0].shape)
# print('Output shape:', baseline(wide_window.example[0]).shape)
# wide_window.plot(model = baseline, plot_col = 'close')

#////////////////////////////////////////

#Linear Model: ************************************************************************
from src.utils.FeatureEngineering.ModelGenerator import ModelGenerator

print()
print('Linear Model Started .............')
print()

wide_window.PriceName = 'high'


linear = tf.keras.Sequential([
								#tf.keras.layers.Flatten(),
								tf.keras.layers.LSTM(32, return_sequences = True),
								tf.keras.layers.Conv1D(
														filters = 32,
							                           	kernel_size = (24,),
							                           	activation = 'relu'),
								#tf.keras.layers.Flatten(),
								tf.keras.layers.Dense(units = 64, activation = 'relu'),
								#tf.keras.layers.LSTM(32, return_sequences = False),
								tf.keras.layers.Dense(units = num_features),
								#tf.keras.layers.Reshape([1, -1]),
							])

print('Input shape:', wide_window.example[0].shape)
print('Output shape:', linear(wide_window.example[0]).shape)

model_generator = ModelGenerator(label_index = column_indices['close_5m'])

history = model_generator.compile_and_fit(model = linear, window = wide_window)

print(history)

val_performance = {}
performance = {}

val_performance['Linear'] = linear.evaluate(wide_window.val)
performance['Linear'] = linear.evaluate(wide_window.test, verbose=0)

wide_window.plot(model = linear, plot_col = 'close_5m')

# import matplotlib.pyplot as plt

# print(linear.layers[0].kernel[:,0].numpy())

# plt.bar(x = range(len(train_df.columns)),
#         height=linear.layers[0].kernel[:,0].numpy())
# axis = plt.gca()
# axis.set_xticks(range(len(train_df.columns)))
# _ = axis.set_xticklabels(train_df.columns, rotation=90)

# plt.show()

#//////////////////


sys.exit()

# print(dataset_5M['XAUUSD_i'].to_timestamp(freq=None, how='start', copy=True))

#*****************************************************************************************************
#Main Features Creation:

# from src.utils.FeatureEngineering.MainFeatures import MainFeatures

# main_features = MainFeatures()
# main_features.symbol = 'XAUUSD_i'
# print('dataset geted')

# dataset = main_features.Get(
# 							symbol = main_features.symbol,
# 							dataset_5M = dataset_5M, 
# 							dataset_1H = dataset_1H,
# 							mode = None
# 							)

# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
# 	print(dataset)

#/////////////////////////////////////////////////////////////////////////////////////////////////////

#*********************************************************************************************************
#LagFeatures:

from src.utils.FeatureEngineering.LagFeatures import LagFeatures
from src.utils.FeatureEngineering.WindowGenerator import WindowGenerator

lagfeature = LagFeatures()
dataset_return = lagfeature.Get(dataset = dataset, symbol = 'XAUUSD_i', number_lags = 6, mode = None)

# print(dataset_return)
# sys.exit()
# print(dataset_return)
# prices, time = dataset_return.index
# print('prices = ', prices)

# print(dataset_return['return_70_1'].dropna())

# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
# print(dataset_return)
# print(dataset_return.loc['close_5m', ['return_70_7', 'real_70']].dropna(subset=['real_70']))

# print(dataset_return.info())
# print(dataset_return.loc['close_5m'])

#//////////////////////////////////////////////////////////////////////////////////////////////

#**************************************************************
#WindowGenerator:

column_indices = {name: i for i, name in enumerate(dataset_return.columns)}
index_indices = {name[0]: i for i, name in enumerate(dataset_return.index)}

# print(index_indices)
print(index_indices.keys())
print(dataset_return[index_indices.keys(),:,:])
n = len(dataset_return.loc['close_5m'])
print(n)
train_df = dataset_return.loc[index_indices.keys()][0:int(n*0.7)]
print(train_df)
val_df = dataset_return.loc['close_5m'][int(n*0.7):int(n*0.9)]
test_df = dataset_return.loc['close_5m'][int(n*0.9):]

num_features = dataset_return.shape[1]

w1 = WindowGenerator(
					input_width=24, 
					label_width=1, 
					shift=24,
					train_df = train_df,
					val_df = val_df,
					test_df = test_df,
                    label_columns=['return_70_7']
                    )

w2 = WindowGenerator(
					input_width=6, 
					label_width=1, 
					shift=1,
					train_df = train_df,
					val_df = val_df,
					test_df = test_df,
                    label_columns=['return_70_7']
                    )

print(w2)

#////////////////////////////////////////////////////////////////////////////////////





#*******************************************************************************************************
# Frequencies Module:

# from src.utils.FeatureEngineering.Frequencies import Frequencies

# frequences = Frequencies()
# frequences = frequences.Get(dataset = dataset, symbol = main_features.symbol, mode = 'Run')
# print(frequences)

#/////////////////////////////////////////////////////////////////////////////////////

#*******************************************************************************************************
# Fourier Feature Module:

# from src.utils.FeatureEngineering.FourierFeatures import FourierFeatures

# fourierfeatures = FourierFeatures()
# fourierfeatures = fourierfeatures.Get(dataset = dataset, symbol = main_features.symbol, mode = None, number_frequencies = 4)
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
# print(fourierfeatures)

#/////////////////////////////////////////////////////////////////////////////////////

#*******************************************************************************************************
#Patterns Module:

# from src.utils.FeatureEngineering.Patterns import Patterns

# patterns = Patterns()
# candle_pattern = patterns.Get(
# 								dataset = dataset, 
# 								mode = 'Run',
# 								dataset_5M = dataset_5M, 
# 								dataset_1H = dataset_1H, 
# 								symbol = main_features.symbol
# 								)

# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
# 	print(candle_pattern)

#/////////////////////////////////////////////////////////////////////////////////////


# dataset = FE.AlphaCandlePatterns(dataset = dataset)
# dataset = FE.AlphaFactorBBAND(dataset = dataset)
# dataset = FE.AlphaFactorSMA(dataset = dataset)
# dataset = FE.AlphaFactorEMA(dataset = dataset)
# dataset = FE.AlphaFactorIchimokou(dataset = dataset)

#FE.FourierCreation(dataset = dataset.resample('20T').last())
# dataset = FE.TimeCreationPattern(dataset = dataset)

# fft = tf.signal.rfft(dataset['close_5m'])


# for elm in fft.numpy():
# 	# print(elm.numpy())
# 	real_num = elm.real
# 	imag_num = elm.imag

# 	if real_num >= 0:
# 		print('real = ', real_num)
# 		print('imag = ', imag_num)
# 		print('abs = ', np.abs(elm))
# 		print()
from src.utils.Optimizers.Optimizers import Optimizers
from src.utils.Optimizers import NoiseCanceller


optimizer = Optimizers()
noise_canceller = NoiseCanceller.NoiseCanceller()

dataset = dataset.drop(columns = ['time_5m'])

# dataset['close_5m'] = noise_canceller.NoiseWavelet(dataset = dataset.reset_index(), applyto = 'close_5m').values
# print(dataset['close_5m'])

freq_time = '5T'
freq = []
freq_out = 1
dataset_filterd = dataset.copy(deep = True)
dataset_filterd['macds_5m_buy_primary'] = np.nan

for i in range(0, 10):

	noise_canceller.scale_haar = 1/freq_out
	noise_canceller.scale_db6 = 1/freq_out
	noise_canceller.scale_dmey = 1/freq_out
	
	dataset_filterd['macds_5m_buy_primary'] = noise_canceller.NoiseWavelet(dataset = dataset.reset_index().copy(deep = True), applyto = 'macds_5m_buy_primary').values

	dataset_freq = dataset_filterd.resample(freq_time).last()

	freq_out, frequences, spectrum = optimizer.FreqFinder(ts = dataset_freq['macds_5m_buy_primary'].dropna())
	freq_time = str(5 * freq_out) + 'T'

	freq.append(freq_out * 5)

	# print(dataset_freq['close_5m'].dropna().index)

	plt.plot(dataset_freq['macds_5m_buy_primary'].dropna().index, dataset_freq['macds_5m_buy_primary'].dropna())
	# print('freq = ', freq_out * 5)

	# plt.step(frequences, spectrum)
	# plt.xscale('log')
	# plt.ylim(0, 1500)
	# plt.xlim([0.1, max(plt.xlim())])
	# plt.xticks([1, 365.2524], labels=['1/Year', '1/day'])
	# _ = plt.xlabel('Frequency (log scale)')
	# print('frequences = ', frequences)
	# print('spectrum = ', spectrum)
plt.show()
print(freq)
fourier_feature = pd.DataFrame()
features = {}


for frequen in freq:
	time = np.arange(len(dataset.index), dtype=np.float32)
	k = 2 * np.pi * (1 / frequen) * time
	
	features.update({
			        f"sin_{frequen}": np.sin(k),
			        f"cos_{frequen}": np.cos(k),
    				})

fourier_feature = pd.DataFrame(features, index = dataset.index)
print(fourier_feature.columns)
plt.plot(dataset.index, fourier_feature['sin_262975'])
plt.plot(dataset.index, fourier_feature['cos_262975'])
plt.show()



# plt.scatter(fft.numpy().real, fft.numpy().imag, marker='*')
# plt.step(frequences, spectrum)
# plt.xscale('log')
# plt.ylim(0, 1500)
# plt.xlim([0.1, max(plt.xlim())])
# plt.xticks([1, 365.2524], labels=['1/Year', '1/day'])
# _ = plt.xlabel('Frequency (log scale)')
# plt.show()
# plt.plot(dataset.index, dataset['close_5m'])
# dataset.index = dataset['time_5m']
# dataset = dataset.resample('15T').last()



for clm in dataset.columns:
	if 'pattern' in clm: continue
	if 'close' in clm: continue
	if 'high' in clm: continue
	if 'low' in clm: continue
	if 'open' in clm: continue
	if 'HL' in clm: continue
	if '1h' in clm: continue

	if 'pattern_day' in clm:
		print(dataset['close_5m'][dataset[clm].dropna().index[0] : ])
		print(dataset[clm].dropna())
		plt.hist2d(dataset['close_5m'][dataset[clm].dropna().index[0] : ], dataset[clm].dropna(), bins=(50, 50), vmax=400)
		plt.colorbar()
		plt.xlabel('close_5m')
		plt.ylabel(clm)
		plt.show()

sys.exit()


# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
# 	print(dataset.describe().transpose())

# DaysOfWeek = ['Monday', 'Tuesday', 'Thursday', 'Wednesday', 'Friday']
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
	
# 	print(dataset[['Monday', 'Tuesday', 'Thursday', 'Wednesday', 'Friday']])

# sys.exit()
dataset = FE.AlphaCandlePatterns(dataset = dataset)
dataset = FE.AlphaDivergencePatternAndFactorOsilator(dataset = dataset, dataset_5M = dataset_5M, dataset_1H = dataset_1H)

# dataset = FE.AlphaFactorBBAND(dataset = dataset)
# dataset = FE.AlphaFactorSMA(dataset = dataset)
# dataset = FE.AlphaFactorEMA(dataset = dataset)
# dataset = FE.AlphaFactorIchimokou(dataset = dataset)

# data = FE.LagCreation(dataset = dataset)
# data = FE.MemontumCreation(dataset = data)
# data = FE.LagShiftedCreation(dataset = data)
# data = FE.TimeCreation(dataset = data)
for clm in dataset.columns:
	with pd.option_context('display.max_rows', None, 'display.max_columns', None):
		print(dataset[clm].dropna().head(5))
print(dataset.info())



data = tf.keras.utils.timeseries_dataset_from_array(
												    data = dataset['close_5m'][:-200].values,
												    targets = dataset['close_5m'][200:].values,
												    sequence_length = 200,
												    sequence_stride = 1,
												    sampling_rate = 1,
												    batch_size = 200,
												    shuffle = False,
												    seed = None,
												    start_index = None,
												    end_index = None,
													)

for batch in data:
	inputs, targets = batch
	assert np.array_equal(inputs[0], dataset['close_5m'][:200].values)  # First sequence: steps [0-9]
	# Corresponding target: step 10
	assert np.array_equal(targets[0], dataset['close_5m'].iloc[200])
	break

print(inputs)
print(targets)

# dataset = FE.AlphaFactorRSI(dataset = dataset)
# dataset = FE.AlphaFactorBBAND(dataset = dataset)
# dataset = FE.AlphaFactorSMA(dataset = dataset)
# dataset = FE.AlphaFactorEMA(dataset = dataset)
# dataset = FE.AlphaFactorIchimokou(dataset = dataset)
# dataset = FE.AlphaFactorMACD(dataset = dataset)
# dataset = FE.AlphaFactorStochAstic(dataset = dataset)
# dataset = FE.AlphaFactorCCI(dataset = dataset)



# data = FE.LagCreation(dataset = dataset)
# data = FE.MemontumCreation(dataset = data)
# data = FE.LagShiftedCreation(dataset = data)
# data = FE.TimeCreation(dataset = data)