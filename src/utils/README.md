### Utils module:

In the Utils module, modules and utilities are defined for various purposes, computing, machine learning, and symbol news.

**1.	DataReader module:**

The DataReader module is responsible for reading information online and offline. In the online mode, the data is read from the MQL5 software with that account set’s in the settings section. It is also possible to save the read data in the robot database by using the functions inside the module and update the price database as often as possible by using the update function.
By using the read function in this module, the price data stored in the database can be read offline from the database as a Panda Dataframe data and used in the codes.

-

**2.	Divergence module:**

The Divergence module has the task of finding all the divergences for the calculated values of the slot that is given to it. Also, a test module has been written inside this module, which is responsible for testing and rating trade’s is done in offline mode. So that the information of the offline trades output from the PRMethod test functions can be given to it, and in the output of the obtained points, the maximum profit or loss, the maximum drawdown can be received.
The Divergence module in the ShadBot package has the ability to check four divergence modes: primary, hidden, buying and selling. In all sections of the codes, the latent divergence is named with a secondary name.

-

**3.	FeatureEngineering module**

The FeatureEngineering module is defined in the ShadBot package for later purposes and versions. This module is currently under development. In this module, the prices are given to the module as a time series. Then features such as Fourier, day, hour, correlation, divergence, trend indicators, oscillator indicators are added to the time series in an engineered way. The purpose of this module is to engineer the data and prepare a time series for use in the MLMethod module. In the future, the output data from the FeatureEngineering module will be entered into TensorFlow and Kears methods in artificial neural networks and deep learning. Then, using artificial neural networks and deep learning, the future price can be forecast with high accuracy and the best buy or sell signal can be found.

-

**4.	ForexNews module:**

The ForexNews module connects to the https://www.forexfactory.com/ site using chromdriver and receives all the daily news related to the symbols and saves it as a .json file in the robot database. The operation of the robot in online trading is such that, when the event of red or orange news for a symbol is close, a trade will not be opened for this symbol from 30 minutes before to 30 minutes after this event. And during this time, the robot just waits until the red and orange news for this symbol period ends.

-

**5.	Optimizers module:**

In the Optimizers module, modules are defined for faster optimization of some parameters.
The NoiseCancller module is responsible for using KalmanFilter or using PyWavelets to filter the values of the prices and remove the noise in the prices. But we know that completely removing the noise in the prices is not a very good thing, because the signals are the result of many of these price noises. In these functions, there are methods, which can be used to preserve the important information of the price signal, as well as remove harmful price noises.

-
