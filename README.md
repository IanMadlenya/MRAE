# Welcome
**This 3-month project aims at detecting and utilizing mean-reverting phenomenon with machine learning techniques.**

The code directory provides everything to perform the implementation of mean-reverting portfolios from scratch, meaning from data retrieval, to machine learning utilization, and strategies set up.

For a matter of size, only the clean file of log-returns appears in the clean data directory. Raw files may be provided upon request.

Other necessary files to generate results are included in the various directories.


# Requirements
* Python 2.7 (see https://www.continuum.io/downloads)
* Keras (see https://keras.io for a step-by-step install guide)
* Mingw

# Scripts description
Scripts are described in the chronological/logical order they have been used
* [marketData](./code/marketData.ipynb)
    * Describe data formatting process, and choices made in the project regarding raw data
    * Standalone notebook which explains step by step process
* [statsDesc](./code/statsDesc.ipynb)
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
* [portfolio_main](./code/portfolio_main.py)
    * Run the above portfolio. Parameters may be edited upond needs ! As for now, parameters in the script are the last parameters retained. See the script to know how to tune parameters.
* [resultsAnalysis_utils](./code/resultsAnalysis_utils.py)
    * A bunch of useful functions to plot results in a fashion way
    * Normally, no need to edit it (except for model improvement)
* [resultsAnalysis_dataLoading](./code/resultsAnalysis_dataLoading.py)
    * Load hard data that are never edited
    * Normally, no need to edit it (except for model improvement)
* [resultsAnalysis_machineLearning](./code/resultsAnalysis_machineLearning.ipynb)
    * Go through neural network results: statistics, plots, correlations, ...
    * Standalone notebook
* [resultsAnalysis_portfolio](./code/resultsAnalysis_portfolio.ipynb)
    * Go through portfolio results: statistics, plots, correlations, ...
    * Standalone notebook

# Results
* Full report may be seen at https://www.dropbox.com/s/gxp7s9w5opchv81/Isnardy_2A_application_rapport.pdf?dl=0
* French abstract may be seen at https://www.dropbox.com/s/9zn0r5ucq04kqzb/Isnardy_2A_application_synthese_FR.pdf?dl=0
* English abstract may be seen at
