# Welcome
**This 3-month project aimed at detecting and utilizing mean-reverting phenomenon with machine learning techniques.**

Code provides everything to perform the implementation of mean-reverting portfolios from scratch, meaning from data retrieval, to machine learning utilization, and strategies set up.

# Preview of results
Analysis were based on 151 financial return series from Euro stocks over the period [2004-2016].
- Below is a typical result from returns series reconstruction using autoencoders.
![alt tag](https://github.com/antisrdy/MRAE/blob/master/images/returnsReconstructedCombined.PNG)
- The following Figure displays returns we got using neural network (autoencoder) buy/sell signals, compared to benchmark.
![alt tag](https://github.com/antisrdy/MRAE/blob/master/images/returns.png)

# Requirements
* Python 2.7 (see https://www.continuum.io/downloads)
* Keras (see https://keras.io for a step-by-step install guide)
* Mingw

# Scripts description
Scripts are described in the chronological/logical order they have been used.
One can play with the whole model just running bold scripts (mains)
* **[marketData](./code/marketData.ipynb)**
    * Describe data formatting process, and choices made in the project regarding raw data
    * Standalone notebook which explains step by step process
* **[statsDesc](./code/statsDesc.ipynb)**
    * Quick statistics on returns. Basics + correlations between stocks
    * Standalone notebook
* [neuralNetwork_utils](./code/neuralNetwork_utils.py)
    * Some utils useful to neural networks: data formatting and plotting functions
    * Normally, no need to edit it (except for model improvement)
* [neuralNetwork_smoothness](./code/neuralNetwork_smoothness.py)
    * Smoothness implementation. See script for further details
    * Normally, no need to edit it (except for model improvement)
* [neuralNetwork](./code/neuralNetwork.py)
    * Describe the neural network structure and all related steps (e.g. fitting, score saving, some plots)
    * Normally, no need to edit it (except for model improvement)
* **[neuralNetwork_main](./code/neuralNetwork_main.py)**
    * Train the above network. Parameters may be edited upond needs !
    * As for now, parameters in the script are parameters retained all along the project. They enable to train all the networks. It is easy just to train a single model: just avoid the loop.
* [portfolio](./code/portfolio.py)
    * Describe a portfolio and the useful flow to implement a strategy
    * Normally, no need to edit it (except for model improvement)
* **[portfolio_main](./code/portfolio_main.py)**
    * Run the above portfolio. Parameters may be edited upond needs ! As for now, parameters in the script are the last parameters retained. See the script to know how to tune parameters.
* [resultsAnalysis_utils](./code/resultsAnalysis_utils.py)
    * A bunch of useful functions to plot results in a fashion way
    * Normally, no need to edit it (except for model improvement)
* [resultsAnalysis_dataLoading](./code/resultsAnalysis_dataLoading.py)
    * Load hard data that are never edited
    * Normally, no need to edit it (except for model improvement)
* **[resultsAnalysis_machineLearning](./code/resultsAnalysis_machineLearning.ipynb)**
    * Go through neural network results: statistics, plots, correlations, ...
    * Standalone notebook
* **[resultsAnalysis_portfolio](./code/resultsAnalysis_portfolio.ipynb)**
    * Go through portfolio results: statistics, plots, correlations, ...
    * Standalone notebook
