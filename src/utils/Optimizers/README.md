**1.	Optimizers module:**

One of the important parameters in divergence signals is the inverse correlation of the oscillator indicator with the price chart (inverse correlation means, correlation < 0). Another important parameter is the number of visible divergence signals between the indicator and the price chart.
Therefore, in this module, the optimizer functions are designed and defined in such a way that in the repeated execution of the functions, it is possible to find the highest inverse correlation of the Oscillator indicator with the price chart, along with the highest number of divergence points.
After finding the divergence points every time, the tested parameters will be stored in the Optimize database corresponding to each indicator in the Indicators module.
This work has been done with two goals:
1- To make the genetic algorithm faster, for the algorithm to converge faster and to find the parameters faster.
2- Use in the FeatureEngineering module as the best parameters with inverse correlation

Also, in this module, the FourierFinder function is defined to find the resonance frequencies of the price, so that it finds the resonance frequencies of the price using FFT, and returns these frequencies and the frequency that creates the most resonance in the prices.

-

**2.	OptimizerRunner module:**

The OptimizerRunner module is responsible for running the optimizers in the ‘Optimizers’ module in multi-threaded parallel processing. Multi-threaded processing is intended to speed up calculations.

-
