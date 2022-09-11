from src.utils.DataReader.MetaTraderReader5.LoginGetData import LoginGetData as getdata
from src.utils.ProtectResist.MLMethod.FeatureEngineering import FeatureEngineering
from src.utils.Optimizers.NoiseCanceller import NoiseCanceller
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import sys
import warnings as warnings
warnings.filterwarnings("ignore")

loging = getdata()

dataset_5M, dataset_1H = loging.readall(symbol = 'XAUUSD_i', number_5M = 'all', number_1H = 2000)

# print(dataset_5M['XAUUSD_i'].to_timestamp(freq=None, how='start', copy=True))

FE = FeatureEngineering()
FE.symbol = 'XAUUSD_i'
print('dataset geted')
dataset = FE.DatasetCreation(dataset_5M = dataset_5M['XAUUSD_i'], dataset_1H = dataset_1H['XAUUSD_i'])
dataset = FE.AlphaDivergencePatternAndFactorOsilator(dataset = dataset, dataset_5M = dataset_5M, dataset_1H = dataset_1H)

# dataset = FE.AlphaCandlePatterns(dataset = dataset)
# dataset = FE.AlphaFactorBBAND(dataset = dataset)
# dataset = FE.AlphaFactorSMA(dataset = dataset)
# dataset = FE.AlphaFactorEMA(dataset = dataset)
# dataset = FE.AlphaFactorIchimokou(dataset = dataset)

#FE.FourierCreation(dataset = dataset.resample('20T').last())
dataset = FE.TimeCreationPattern(dataset = dataset)

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