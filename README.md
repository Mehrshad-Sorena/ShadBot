![ShadRad Electronic](https://www.shadrad.ir/wp-content/uploads/2021/11/Shadrad1-1.png)
Forex Robot
-

### 1- ShadBot package:

The ShadBot package is made in several subsections. All codes are written in Python. The ShadBot package has the ability to conduct trade online, perform backtests for offline trades, and review and analyze them, as well as optimize trades. Also, this package has the ability to optimize using artificial intelligence and predict future price values using machine learning algorithms.
In this package, Scipy and Fitter packages are used to perform mathematical calculations, statistics and probabilities such as normal distributions. Also, the ScikitLearn package is used for the machine learning part, which is mostly Kmeans and regression functions. Under the other parts of this package, the genetic algorithm for AI is written, so that parameters and indicators can be optimized with genetics to find the best mode for trading in the forex market. Also, in one of the sub-sections of this package, a module named OnlineBot is defined, which is an online trading robot of the ShadBot package. so that it can do online trades  in MQL5 software. In another module of the subsections of the ShadBot package, genetic algorithm runners for different modes and three indicators MACD, RSI and Stochastic are written, by runing each of these functions, the parameters of that indicator can be optimized with genetics.
The main module defined as a subsection of the ShadBot package is named Src, and the Indicators and Utils modules are available under this module. In the main part of the ShadBot package, several Main functions are considered for different executions, so that one of them can be processed in parallel in multi-threaded processes for the online trader to perform online and real trades according to the account that It is set in the robot settings. Other executors in the package are currently defined to perform genetic algorithm optimizations.

-
**1-2- OnlineBot:**

In this module, there are robot runners for online trades. So that the runners are obliged to call the online traders related to each indicator, to receive the signals related to the divergence strategy, if the negative red and orange news in the news of the trading day is not the desired symbol, with the calculations of the amount Tradable Lot, as well as checking the basket manager, and checking the risk of trades, transfer the transactions to the MQL5 software using Carrier functions and perform online trades. Also, in this reunner, the codes are written in such a way that all the functions are executed in a multi-threaded, simultaneous and parallel manner.

-
**1-3- __mainbot__:**

This function is responsible for executing all the executors inside the OnlineBot module and executes all the functions at 5 minute intervals. And in the 5M timeframe, it checks all divergence signals. The information related to the news of each symbol will be stored in the robot database on a daily basis.

-
**1-4- __mainRunner__:**

Functions defined with names similar to this are: __mainRunnerBuyPrimary__, __mainRunnerBuySecondry__, __mainRunnerSellPrimary__, __mainRunnerSellSecondry__
Each of these functions is responsible for performing genetic algorithm optimization as well as checking permit for online trades for each of the individual divergence states. So that by implementing each one, you can optimize all three indicators MACD, RSI and Stochastic.
It should be noted that, throughout this package, secondary divergence refers to hidden divergence.

-

### Quick Project Setup
1. Run `sudo git clone git@github.com:Mehrshad-Sorena/ShadBot.git`
2. Install Metatrader5 from `https://www.metatrader5.com/en/download`

-

**Packages:**

3. Install below packages:

```
pip install pandas-ta
python -m pip install --user numpy scipy matplotlib ipython jupyter pandas sympy nose
pip install -U scikit-learn
sudo snap install tqdm
pip install --upgrade mplfinance
sudo yum -y install progress
pip install schedule
pip install shapely==1.8.1.post1
pip install fitter
pip install lxml
```

**For linux OS:**

1.Install Wine.

2.Install Python for Windows on Linux.

3.Find the path to python.exe:

I installed on `/home/user/.wine/drive_c/users/user/Local Settings/Application Data/Programs/Python/Python39`.

4.Install mt5 library on your Windows Python version:

```
pip install MetaTrader5
pip install --upgrade MetaTrader5
```

5.Install this package on your Linux Python version:

```
pip install mt5linux
```