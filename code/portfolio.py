
# coding: utf-8

# # Portfolio strategy

# In[ ]:

from bokeh.io import gridplot, output_file, output_notebook
from bokeh.models import Span
from bokeh.palettes import Spectral7
from bokeh.plotting import figure, output_file, show, reset_output, save
from pandas.tseries.offsets import *

import datetime
import itertools
import logging
import numpy as np
import os
import pandas as pd
import neuralNetwork_utils as ut
import warnings
warnings.filterwarnings('ignore')


# ## Data frames handling
# - Predictions and real values to run strategies
# - Market data to keep track of daily prices

# In[ ]:

# Data split
## LB: lower bound
## UB: upper bound
trainUB = '2012-12'
validationLB = '2013-1'
validationUB = '2013-12'
testLB = '2014'


# ### Hard data
# No need to reload them each time. No matter the model used, they're real data

# In[ ]:

# Keep track of daily prices, useful while trading
pricesPath = '../donnees/clean/PX_LAST.csv'
prices = ut.getInputs(scalingFactor=1,
                      stocksFile=pricesPath)


# In[ ]:

# Real values to be compared with neural network outputs
returnsPath = '../donnees/clean/RET_PX_LAST.csv'
returns = ut.getInputs(scalingFactor=1,
                       stocksFile=returnsPath)
train, validation, test = ut.splitData(returns,
                                       trainUB=trainUB,
                                       validationLB=validationLB,
                                       validationUB=validationUB,
                                       testLB=testLB)


# ### Light data
# Depends on selected model

# In[ ]:

# Predictions to be compared with real values.
def loadPredictions(predictionsPath):
    predictions = ut.getInputs(scalingFactor=1,
                               stocksFile=predictionsPath)
    return predictions


# ## Finance section
# - Portfolio
#     - Global portfolio definition
#     - Portfolio specifications
#         - Uniform portfolio
#         - Risk contribution fashion portfolio
#     - Stock
#     
# - Strategy
#     - Update outstanding portfolio
#     - Creation of new portfolio

# ### Portfolio definition

# In[ ]:

class Portfolio(object):
    def __init__(self,
                 amount, # Daily money spent on trading
                 involvedProportion, # In case of ERC allocation, on average, how much does an action is involved in trading?
                 isLong, # Is there a long leg?
                 isShort, # Is there a short leg?
                 legSize, # In case of fixed leg size
                 legSizeLong, # In case of fixed leg size
                 legSizeShort, # In case of fixed leg size
                 longShortMinLegSize, # In case of unfixed leg size, impose a minimum
                 predictionsTest, # Utils
                 stocksInit, # Initial stocks: all stocks in the universe
                 strategyLength, # How many days does the strategy hold positions?
                 thresholds, # In case of ERC allocation, stock thresholds
                 tracker, # Logging purpose
                 tradingDay): # Log when positions are taken
        
        # ATTRIBUTES
        ## Actors: in this portfolio, who is doing what?
        self.buyers = []
        self.sellers = []
        
        ## Actions selection purpose
        self.legSize = legSize
        self.legSizeLong = legSizeLong
        self.legSizeShort = legSizeShort
        self.isLong = isLong
        self.isShort = isShort
        
        ## Dates
        self.allInDay = None # Day positions are left
        self.tradingDay = tradingDay # Day positions are taken (= day Portfolio instance is instanciated)
        
        ## Init with all stocks in the universe
        ##### GLOBAL (shared) variable #####
        self.stocks = {}
        for stockName, initValue in stocksInit.iteritems():
            self.stocks[stockName] = Stock(stockName=stockName)
            
        ## "Money"
        ### Daily money spent in each leg
        self.amount = amount
        
        ## Portfolio metrics
        self.costs = 0
        self.netValue = 0
        self.rawValue = 0
        
        ## Portfolio strategy
        self.involvedProportion = involvedProportion
        self.isTradingHappening = False # Default behavior
        self.longShortMinLegSize = longShortMinLegSize
        self.strategyLength = strategyLength # How many days positions are held? Day2day by default, maximum 4-5 days (week)
        self.thresholds = thresholds # Not defined in case od audited (and constrained) portfolio
        self.weights = {} # Portfolio allocation
        
        # Predictions to be compared with real data
        self.predictionsTest = predictionsTest
        
        # Track daily performance
        self.tracker = tracker
        
    def addBuyer(self, buyer):
        self.buyers.append(buyer)
        return True
        
    def addSeller(self, seller):
        self.sellers.append(seller)
        return True
        
    def computeCosts(self):
        pass # See subclasses
    
    def computeNetValue(self):
        self.netValue = self.netValue - self.costs
        return True
    
    def computeRawValue(self):
        """
        Retrieve raw value.
        
        2 cases:
        
        - Either it is a long short portfolio.
          Then raw value is computed "naturally', meaning what's been bought on trading day is sold on all in day.
          Earnings account for available money (or loss depending on the sign).
          Inverse process takes place for what's been sold.
          Raw value is then the difference.
        
        - Or it is a long or short only portfolio.
          Same trading operations as before occur (for one of the long/short legs).
          But earnings must be compared to invested amount
        """
        # When called, stocks in portfolio are only stocks involved in trading
        rawValue = 0
        for stockName, stock in self.stocks.iteritems():
            # Common values
            dealValue = stock.nbStock * stock.dayPrice # Daily deal (either B or S)
            self.tracker.ix[pd.to_datetime(self.tradingDay), (stockName, 'valueOut')] = dealValue # log
            gain = dealValue - self.tracker.ix[pd.to_datetime(self.tradingDay), (stockName, 'valueIn')]
            
            # Bought on trading day, sold on all in day
            if stockName in self.buyers:
                self.tracker.ix[pd.to_datetime(self.tradingDay), (stockName, 'gain')] = gain # log

                # Reaching this point implicitly means it is a long portfolio either way
                if self.isShort: rawValue += dealValue # First case: long short portfolio
                elif not self.isShort: rawValue += gain # Second case: long only portfolio. 
                else: print 'Problem' # Shouldn't occur
            
            # Sold on trading day, bought on all in day
            elif stockName in self.sellers:
                # Log minus gain because what stands for the real gain
                # is the remaining money from the sell on trading day
                self.tracker.ix[pd.to_datetime(self.tradingDay), (stockName, 'gain')] = - gain # log
                
                # Reaching this point implicitly means it is a short portfolio either way
                if self.isLong: rawValue -= dealValue # First case: long short portfolio
                elif not self.isLong: rawValue += - gain # Second case: short only portfolio
                else: print 'Problem' # Shouldn't occur

            else: print 'Problem' # Shouldn't occur
                
        self.rawValue = rawValue
        
        return True
        
    def computeWeights(self):
        """
        2 policies:
         - Either leg size is imposed (no matter the kind of portfolio).
           It means some buyers and/or some sellers have been picked up for sure => No special condition.
           
         - Or leg size is not imposed and it evolves "naturally".
           It means some days may miss buyers and/or sellers in case of long short portfolio.
           => Need to handle it and avoid single leg in case of long short portfolio
        """
        if self.legSize:
            # Make sure weights computation is working properly in any case (longshort, long only, short only)
            if ((self.isLong and self.isShort and # Long short
                 self.setWeights(self.buyers) and
                 self.setWeights(self.sellers)) or
                (self.isLong and not self.isShort and self.setWeights(self.buyers)) or # Long only
                (self.isShort and not self.isLong and self.setWeights(self.sellers))): # Short only
                return self.updateWeightAttributes()
            else: return False
            
        elif not self.legSize:
            # Make sure weights computation is working properly in any case (longshort, long only, short only)
            # In case of long short portfolio, impose a condition on minimal number of actions involved
            if ((self.isLong and self.isShort and # Long short
                 len(self.buyers) >= self.longShortMinLegSize and len(self.buyers) >= self.longShortMinLegSize and
                 self.setWeights(self.buyers) and
                 self.setWeights(self.sellers)) or
                (self.isLong and not self.isShort and self.setWeights(self.buyers)) or # Long only
                (self.isShort and not self.isLong and self.setWeights(self.sellers))): # Short only
                return self.updateWeightAttributes()
            else: return False
        else: return False
        return False
        
    def dailyUpdate(self, day):
        self.strategyLength -= 1 # Decrease timer
        if self.strategyLength == 0:
            self.goAllIn(day=day) # Time to go all in
            return True
        else: return False
        
    def getDailyStocksToTrade(self):
        """
        Retrieve daily stocks to trade.
        
        2 policies:
        
        - Either let the system evolve "naturally", based on pre-computed thresholds.
          It means that only actions whose error (epsilon) is above/under pre-computed thresholds
          are part of the trading day.
          
        - Or impose the number of trading actions in each leg.
          It means pick up actions with the largest RELATIVE epsilons (positive and/or negative).
        
        3 cases in each policy:
        
        - long only
        
        - short only
        
        - long short
        """
        dday = pd.to_datetime(self.tradingDay)
        
        # epsilon = y - y_hat
        ##### GLOBAL (shared) variable #####
        epsilon = test - self.predictionsTest
        
        # Policy 1: number of trading actions per leg NOT imposed
        if not self.legSize:
            for stockName, stock in self.stocks.iteritems():
                epsilonDayStock = epsilon.ix[dday, stockName]
                
                # Buy if:
                # - longshort or long only portfolio (isLong = True)
                # - epsilon is negative enough (return_real << return_predicted)
                if self.isLong and epsilonDayStock < self.thresholds.ix[self.involvedProportion / 2, stockName]:
                    stock.setEpsilon(epsilon=epsilonDayStock)
                    self.addBuyer(stockName)
                    
                # Sell if:
                # - longshort or short only portfolio (isShort = True)
                # - epsilon is positive enough (return_real >> return_predicted)
                if self.isShort and epsilonDayStock > self.thresholds.ix[1 - self.involvedProportion / 2, stockName]:
                    stock.setEpsilon(epsilon=epsilonDayStock)
                    self.addSeller(stockName)
                
        # Policy 2: number of trading actions per leg imposed
        elif self.legSize: 
            epsilonDay = epsilon.loc[dday]
            
            epsilonDayRelative = epsilonDay / test.loc[dday]
            epsilonDayRelative = epsilonDayRelative.sort_values(inplace=False)            
            epsilonDayRelative = epsilonDayRelative.replace([np.inf, -np.inf], np.nan).dropna()
            
            # Buyers stand for smallest relative epsilons
            if self.isLong: self.buyers = list(epsilonDayRelative[:self.legSizeLong].keys())
                
            # Sellers stand for largest relative epsilons
            if self.isShort: self.sellers = list(epsilonDayRelative[-self.legSizeShort:].keys())
        
        else:
            print('Problem')
            return False
        
        # Discard non-trading stocks
        for stock in set(self.stocks.keys()).difference(set(self.buyers + self.sellers)): del self.stocks[stock]
        
        return True
    
    def goAllIn(self, day):
        # Update attributes
        self.setAllInDay(allInDay=day)
        for stockName, stock in self.stocks.iteritems():
            ##### GLOBAL (shared) variable #####
            stock.setDayPrice(dayPrice=prices.ix[day, stockName])  
            
        # Track performance
        self.computeRawValue()
        self.computeCosts()
        self.computeNetValue()
        
        return True
    
    def setAllInDay(self, allInDay):
        self.allInDay = allInDay
        return True
        
    def setIsTradingHappening(self, isTradingHappening):
        self.isTradingHappening = isTradingHappening
        return True
            
    def setTradingDay(self, day):
        self.tradingDay = day
        return True
    
    def setWeights(self):
        pass # See subclasses
            
    def trade(self):
        for stockName, stock in self.stocks.iteritems():
            # Keep track of stock price
            ##### GLOBAL (shared) variable #####
            stock.setDayPrice(dayPrice=prices.ix[self.tradingDay, stockName])
            ddayStockPrice = stock.dayPrice
            
            dealValue = self.amount * np.abs(stock.weight)
            self.tracker.ix[pd.to_datetime(self.tradingDay), (stockName, 'valueIn')] = dealValue
            
            # No matter long/short position:
            nbStock = dealValue / ddayStockPrice
            stock.setNbStock(nbStock=nbStock)
            
        return True
    
    def updateWeightAttributes(self):
        for stock in self.sellers: self.weights[stock] = - self.weights[stock]
        for stock in self.buyers + self.sellers: # log
            self.tracker.ix[pd.to_datetime(self.tradingDay), (stock, 'weight')] = self.weights[stock]
        self.setIsTradingHappening(isTradingHappening=True)
        
        return True


# ### Portfolio specifications
# 
# Override weight allocation method

# In[ ]:

class RiskContributionPortfolio(Portfolio):
    def __init__(self,
                 amount,
                 involvedProportion,
                 isLong,
                 isShort,
                 legSize,
                 legSizeLong,
                 legSizeShort,
                 longShortMinLegSize,
                 predictionsTest,
                 stocksInit,
                 strategyLength,
                 thresholds,
                 tracker,
                 tradingDay):
        super(RiskContributionPortfolio, self).__init__(amount,
                                                        involvedProportion,
                                                        isLong,
                                                        isShort,
                                                        legSize,
                                                        legSizeLong,
                                                        legSizeShort,
                                                        longShortMinLegSize,
                                                        predictionsTest,
                                                        stocksInit,
                                                        strategyLength,
                                                        thresholds,
                                                        tracker,
                                                        tradingDay)
        
        self.weight_type = 'riskContribution'
    
    def setWeights(self, who):
        if len(who) > 0:
            end = pd.to_datetime(self.tradingDay)
            start = end - DateOffset(months=12) # Take last year

            returnsRescaled = returns[start:end][who]

            vols = returnsRescaled.std()
            volsInv = 1 / vols
            sumVolsInv = volsInv.sum()

            for stock in who:
                weight = volsInv[stock] / sumVolsInv
                self.stocks[stock].setWeight(weight)
                self.weights[stock] = weight
            return True
        else: return False

class UniformPortfolio(Portfolio):
    def __init__(self,
                 amount,
                 involvedProportion,
                 isLong,
                 isShort,
                 legSize,
                 legSizeLong,
                 legSizeShort,
                 longShortMinLegSize,
                 predictionsTest,
                 stocksInit,
                 strategyLength,
                 thresholds,
                 tracker,
                 tradingDay):
        super(UniformPortfolio, self).__init__(amount,
                                               involvedProportion,
                                               isLong,
                                               isShort,
                                               legSize,
                                               legSizeLong,
                                               legSizeShort,
                                               longShortMinLegSize,
                                               predictionsTest,
                                               stocksInit,
                                               strategyLength,
                                               thresholds,
                                               tracker,
                                               tradingDay)
        
        self.weight_type = 'uniform'
    
    def setWeights(self, who):
        if len(who) > 0:
            weight = 1. / len(who)
            for stock in who:
                self.stocks[stock].setWeight(weight)
                self.weights[stock] = weight
            return True
        else: return False


# ### Stock definition

# In[ ]:

class Stock:
    def __init__(self, stockName):
        self.dayPrice = None
        self.epsilon = 0
        self.name = stockName
        self.nbStock = 0
        self.weight = -1
        
    def setDayPrice(self, dayPrice):
        self.dayPrice=dayPrice
        
    def setEpsilon(self, epsilon=-1):
        self.epsilon = epsilon
            
    def setNbStock(self, nbStock):
        self.nbStock = nbStock
    
    def setWeight(self, weight):
        self.weight = weight


# ### Thresholds

# In[ ]:

class Thresholds:
    def __init__(self,
                 involvedProportion,
                 predictionsTrain):
        
        self.involvedProportion = involvedProportion
        self.predictionsTrain = predictionsTrain
        
    def gatherThresholds(self):
        ##### GLOBAL (shared) variable #####
        residuals = train - self.predictionsTrain
        quantiles = residuals.quantile([self.involvedProportion / 2, 1 - self.involvedProportion / 2])
        return quantiles


# ### Strategy definition
# - Run strategy, including:
#     - Update current portfolios
#     - Create a new portfolio on a daily-basis

# In[ ]:

class Strategy:
    def __init__(self,
                 amount,
                 involvedProportion,
                 isLong,
                 isShort,
                 legSize,
                 legSizeLong,
                 legSizeShort,
                 longShortMinLegSize,
                 predictionsTest,
                 stocksInit,
                 strategyLength,
                 thresholds,
                 weight_type):
        
        # Track daily performance
        ##### GLOBAL (shared) variable #####
        iterables = [stocksInit.keys(), ['weight',
                                         'valueIn',
                                         'valueOut',
                                         'gain']]
        index = pd.MultiIndex.from_product(iterables, names=['Action', 'Values'])
        self.tracker = pd.DataFrame(index=['-1'], columns=index)
        #####
        
        ##### GLOBAL (shared) variable #####
        self.period = test.index.get_values()
        
        # Strategy specs
        self.amount = amount
        self.involvedProportion = involvedProportion
        self.isLong = isLong
        self.isShort = isShort
        self.legSize = legSize
        self.legSizeLong = legSizeLong
        self.legSizeShort = legSizeShort
        self.longShortMinLegSize = longShortMinLegSize
        self.predictionsTest = predictionsTest
        self.stocksInit = stocksInit
        self.strategyLength = strategyLength
        self.thresholds = thresholds
        self.weight_type = weight_type
    
    def dailyUpdates(self,
                     day,
                     portfoliosToTrack):
        '''
        If maturity is reached: go all in and take reverse positions
        '''
        ##### Previous portfolio(s) #####
        # Perform portfolio updates and portfoliosToTrack updates
        for portfolioToTrack in portfoliosToTrack:
            # True means the portfolio has gone all in
            if portfolioToTrack.dailyUpdate(day): portfoliosToTrack.remove(portfolioToTrack)
            else: pass # Nothing to do, except above update
        return True

    def dailyCreation(self,
                      day):
        '''
        Upon instanciation:
        - Find stocks to be traded
        - Compute weights
        - Trade    
        '''
        # New portfolio
        ## Init new portfolio
        portfolio = None
        if self.weight_type == 'uniform': portfolio = UniformPortfolio(amount=self.amount,
                                                                       involvedProportion=self.involvedProportion,
                                                                       isLong=self.isLong,
                                                                       isShort=self.isShort,
                                                                       legSize=self.legSize,
                                                                       legSizeLong=self.legSizeLong,
                                                                       legSizeShort=self.legSizeShort,
                                                                       longShortMinLegSize=self.longShortMinLegSize,
                                                                       predictionsTest=self.predictionsTest,
                                                                       stocksInit=self.stocksInit,
                                                                       strategyLength=self.strategyLength,
                                                                       thresholds=self.thresholds,
                                                                       tracker=self.tracker,
                                                                       tradingDay=day)

        elif self.weight_type == 'riskContribution':
            portfolio = RiskContributionPortfolio(amount=self.amount,
                                                  involvedProportion=self.involvedProportion,
                                                  isLong=self.isLong,
                                                  isShort=self.isShort,
                                                  legSize=self.legSize,
                                                  legSizeLong=self.legSizeLong,
                                                  legSizeShort=self.legSizeShort,
                                                  longShortMinLegSize=self.longShortMinLegSize,
                                                  predictionsTest=self.predictionsTest,
                                                  stocksInit=self.stocksInit,
                                                  strategyLength=self.strategyLength,
                                                  thresholds=self.thresholds,
                                                  tracker=self.tracker,
                                                  tradingDay=day)
        else: return False
        
        # Perform operations
        ## Who is trading today?
        portfolio.getDailyStocksToTrade()

        ## Weights
        if not portfolio.computeWeights(): return(portfolio) # False means no weight has been computed: no trading today

        ## Trading operations
        portfolio.trade()

        return(portfolio)

    def run(self):
        '''
        Run strategy over the full test period
        '''        
        portfolios = []
        portfoliosToTrack = []

        for day in self.period:
            self.dailyUpdates(day=day,
                              portfoliosToTrack=portfoliosToTrack)

            portfolio = self.dailyCreation(day=day)

            portfolios.append(portfolio)

            if portfolio.isTradingHappening: portfoliosToTrack.append(portfolio)

        return(portfolios)


# ### Strategy performance
# Utils mainly

# In[ ]:

def getValues(portfolios): # portfolios is a list of portfolios
    # Raw value for now; will change after costs taken into account
    absValues = [portfolio.rawValue for portfolio in portfolios]
    relValues = np.cumsum(absValues) # Cumul
    dates = [portfolio.tradingDay for portfolio in portfolios]
    return(absValues, relValues, dates)


# In[ ]:

def getMetrics(portfolios):
    absValues, _, _ = getValues(portfolios)
    absValuesWO0 = [val for val in absValues if val != 0]
    return(np.mean(absValuesWO0), np.std(absValuesWO0))


# In[ ]:

def saveValues(portfolios, parametersSetML, parametersSetStrat, outputDir='../results/dae/portfolios/values/'):
    portfolioAbsValues, portfolioRelValues, dates = getValues(portfolios)
    dictToDF = {'AbsGain': portfolioAbsValues,
                'CumulGain': portfolioRelValues,
                'Date': dates}
    res = pd.DataFrame(dictToDF)
    path = outputDir + parametersSetML + '__' + '_'.join(parametersSetStrat) + '.csv'
    res.to_csv(path_or_buf=path,
               sep=';')
    return True


# In[ ]:

# Plot daily value of portfolio
def plotPortfoliosPerf(portfolios,
                       parametersSetML,
                       parametersSetStrat,
                       outputDir='../results/dae/portfolios/perf/'): # Portfolios of portfolios actually
    reset_output()
    output_file(outputDir + parametersSetML + '__' + '_'.join(parametersSetStrat) + '.html')
    
    strategiesLength = range(1, len(portfolios) + 1)
    
    portfolioAbsValues = []
    portfolioRelValues = []
    for portfolio in portfolios: # portfolio is a list of portfolios
        absValues, relValues, _ = getValues(portfolio)
        portfolioAbsValues.append(absValues)
        portfolioRelValues.append(relValues)
    
    # Set up plot
    grid = []
    
    # Cumul plot
    strategiesLength = [str(strategyLength) for strategyLength in strategiesLength]
    numlines=len(portfolios)
    colors_list=Spectral7[0:numlines]
    involvedProportion = portfolios[0][0].involvedProportion
    title = 'Cumulative performances for involved proportion ' + str(involvedProportion)# + ' threshold: ' #+ threshold
    p1 = figure(title=title,
                x_axis_type="datetime",
                background_fill_color="#E8DDCB")
    
    xs=[test.index.values]*numlines ## Global var
    ys=[portfolioRelValues[i] for i in range(numlines)]
    
    for (colr, leg, x, y ) in zip(colors_list, strategiesLength, xs, ys):
                p1.line(x, y, color=colr, legend=leg)
    p1.legend.orientation = "bottom_right"
    
    hline = Span(location=0, dimension='width', line_color='black', line_width=2)
    p1.renderers.extend([hline])
    
    # Gain distribution
    p2 = figure(title='Gain distribution', background_fill_color="#E8DDCB")
    portfolioAbsValues = portfolioAbsValues[0] # [0] because for now, just interested in single strategyLength !! CAUTION
    hist, edges = np.histogram([val for val in portfolioAbsValues if val != 0], density=True, bins=40)
    p2.xaxis.axis_label = 'Strategy returns'
    p2.yaxis.visible = None
    p2.xaxis.axis_label_text_font_size = "12pt"
    p2.quad(top=hist,
            bottom=0,
            left=edges[:-1],
            right=edges[1:],
            fill_alpha=0.5,
            fill_color='green')
    
    grid.append([p1, p2])
    p = gridplot(grid)
    save(p)
    return True


# In[ ]:

def logBuyersSellers(portfolios, params):
    sbLogger = logging.getLogger()
    sbLogger.setLevel(logging.INFO)
    sbHandler = logging.FileHandler('../results/dae/portfolios/bs/bs_' + params + '.log', mode='w')
    sbHandler.setLevel(logging.INFO)
    sbFormatter = logging.Formatter('%(message)s')
    sbHandler.setFormatter(sbFormatter)
    sbLogger.addHandler(sbHandler)
    
    for portfolio in portfolios:
        sbLogger.info(portfolio.tradingDay)
        sbLogger.info("%s", portfolio.weights)
    
    sbHandler.close()
    sbLogger.removeHandler(sbHandler)
    del sbHandler, sbLogger
    return True


# # Main

# In[ ]:

def main(params):
    
    (amount,
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
     weight_type) = params
    
    # Track parameters
    parametersSetML = predictionsPathTest.rsplit('/')[-1].replace('.csv', '').replace('test_', '')
    
    parametersSetStrat = [amount,
                          strategyLength,
                          isLong,
                          isShort,
                          legSize,
                          legSizeLong,
                          legSizeShort,
                          involvedProportion,
                          longShortMinLegSize,
                          weight_type]
    parametersSetStrat = [str(param) for param in parametersSetStrat]
    
    # Track predictions
    predictionsTest = loadPredictions(predictionsPath=predictionsPathTest)
    predictionsTrain = loadPredictions(predictionsPath=predictionsPathTrain)
    
    # Gather relative thresholds
    thresholds = None
    if legSize == False:
        thresholdsInstance = Thresholds(involvedProportion=involvedProportion,
                                        predictionsTrain=predictionsTrain)
        thresholds = thresholdsInstance.gatherThresholds()
    
    # Launch strategy
    strategy = Strategy(amount=amount,
                        involvedProportion=involvedProportion,
                        isLong=isLong,
                        isShort=isShort,
                        legSize=legSize,
                        legSizeLong=legSizeLong,
                        legSizeShort=legSizeShort,
                        longShortMinLegSize=longShortMinLegSize,
                        predictionsTest=predictionsTest,
                        stocksInit=stocksInit,
                        strategyLength=strategyLength,
                        thresholds=thresholds,
                        weight_type=weight_type)
    portfolios = strategy.run()
    
    concatConf = parametersSetML + '__' + '_'.join(parametersSetStrat)
    
    # Compute KPIs
    mean, std = getMetrics(portfolios)
    
    perfLogger.info('%s;%f;%f', concatConf, mean, std)
    
    logBuyersSellers(portfolios=portfolios, params=concatConf)
    
    plotPortfoliosPerf(portfolios=[portfolios],
                       parametersSetML=parametersSetML,
                       parametersSetStrat=parametersSetStrat)
    
    saveValues(portfolios=portfolios,
               parametersSetML=parametersSetML,
               parametersSetStrat=parametersSetStrat)
    
    strategy.tracker[1:].to_csv('../results/dae/portfolios/trackers/' + concatConf + '.csv',
                                sep=';',
                                index_label='Date')
    
    return True

