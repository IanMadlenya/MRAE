
# coding: utf-8

# In[ ]:

from portfolio import *


# In[ ]:

if __name__ == "__main__":
    
    # TO BE TUNED: DESCRIBE WHICH KIND OF OPERATION IT IS
    operationTitle = 'xxx'
    
    # Performance logger
    perfLogger = logging.getLogger('perf')
    perfLogger.setLevel(logging.INFO)
    ## create a file perfHandler
    perfHandler = logging.FileHandler('../results/dae/portfolios/performances_portfolios_' + operationTitle + '.csv', mode='w')
    perfHandler.setLevel(logging.INFO)
    ## create a logging format
    perfFormatter = logging.Formatter('%(message)s')
    perfHandler.setFormatter(perfFormatter)
    ## add the perfHandlers to the perfLogger
    perfLogger.addHandler(perfHandler)
    perfLogger.info('specs;mean;std')
    
    # Hyperparameters
    ## Initialization
    stocksInit = {}
    for i, stockName in enumerate(returns.columns.values):
        stocksInit[stockName] = 0
    
    amountInvested = 100000
    
    refDir = '../results/dae/neuralNetwork/predictions/'
    
    # TO BE TUNED: WHICH MODEL(S) IS USED TO TRAIN STRATEGIES
    confs = ['10_75_False_None_adadelta_mse_200_10_tanh_linear']
    
    for conf in confs:
        predictionsPathTest = refDir + 'test_' + conf + '.csv'
        predictionsPathTrain = refDir + 'train_' + conf + '.csv'

        # TO BE TUNED (values after # are alternative values that could be used)
        involvedProportion = None
        #[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        # (in that case, need to implement an additional loop below)
        isLong = True # False
        isShort = True # False
        legSize = True # False: in that case, involved is not None
        legSizeLong = 1 # None, 2, 3,...
        legSizeShort = 1 # None, 2, 3,...
        longShortMinLegSize = None # 1
        strategyLength = 1 # 2, 3, ... How many days positions are held
        weight_type = 'riskContribution' # uniform
        
        params = (amountInvested,
                  involvedProportion,
                  isLong,
                  isShort,
                  legSize,
                  legSizeLong,
                  legSizeShort,
                  longShortMinLegSize,
                  perfLogger,
                  predictionsPathTest,
                  predictionsPathTrain,
                  stocksInit,
                  strategyLength,
                  weight_type)

        main(params)
        
    perfHandler.close()
    perfLogger.removeHandler(perfHandler)
    del perfHandler, perfLogger

