
# coding: utf-8

# # Results analysis: data loading
# 
# Load hard data that are never edited

# In[ ]:

import pandas as pd
import neuralNetwork_utils as ut


# In[ ]:

trainUB = '2012-12'
validationLB = '2013-1'
validationUB = '2013-12'
testLB = '2014'


# In[ ]:

returnsPath = '../donnees/clean/RET_PX_LAST.csv'
returns = ut.getInputs(scalingFactor=1,
                       stocksFile=returnsPath)
train, validation, test = ut.splitData(returns,
                                       trainUB=trainUB,
                                       validationLB=validationLB,
                                       validationUB=validationUB,
                                       testLB=testLB)


# In[ ]:

parametersSetML = ['scalingFactor',
                   'encoding_dim',
                   'denoising',
                   'dropoutProba',
                   'optimizer',
                   'loss_func',
                   'nb_epoch',
                   'batch_size',
                   'activFirstLayer',
                   'activSecondLayer']

numericML = ['scalingFactor',
             'encoding_dim',
             'dropoutProba',
             'nb_epoch',
             'batch_size',
             'loss',
             'smoothness']

parametersSetStrat = ['amount',
                      'strategyLength',
                      'isLong',
                      'isShort',
                      'legSize',
                      'legSizeLong',
                      'legSizeShort',
                      'involvedProportion',
                      'longShortMinLegSize',
                      'weight_type']

numericStrat = ['amount',
                'strategyLength',
                'legSizeLong',
                'legSizeShort',
                'involvedProportion',
                'longShortMinLegSize']


# In[ ]:

print '----------'
print 'Date partition :'
print ''
print ' - train upper bound : %s'%trainUB
print ' - validation : %s - %s'%(validationLB, validationUB)
print ' - test lower bound : %s'%testLB
print ''
print 'Available variables :'
print ''
print ' - returns'
print ' - train'
print ' - validation'
print ' - test'
print ''
print ' - parametersSetML :'
print '   %s'%parametersSetML
print ' - numericML :'
print '   %s'%numericML
print ' - parametersSetStrat :'
print '   %s'%parametersSetStrat
print ' - numericStrat :'
print '   %s'%numericStrat

print '----------'

