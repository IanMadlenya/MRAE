
# coding: utf-8

# # Neural network
# 
# - Implement denoising autoencoder
# - Track metrics

# In[ ]:

from keras import regularizers
from keras.layers import Input
from keras.layers.core import Dense
from keras.models import Model
from neuralNetwork_smoothness import measureSmoothness

import itertools
import logging
import numpy as np
import scipy as sp
import pandas as pd
import neuralNetwork_utils as ut


# ## Neural Network core

# In[ ]:

class BasicAutoencoder:
    def __init__(self,
                 scalingFactor,
                 encoding_dim,
                 denoising,
                 dropoutProba,
                 optimizer,
                 loss,
                 nb_epoch,
                 batch_size,
                 activFirstLayer,
                 activSecondLayer,
                 trainUB,
                 validationLB,
                 validationUB,
                 testLB):
        
        # Neural network parameters
        ## Input
        self.scalingFactor = scalingFactor
        self.trainUB = trainUB
        self.validationLB = validationLB
        self.validationUB = validationUB
        self.testLB = testLB
        self.stockNames = None
        
        self.x_train = None
        self.y_train = None
        self.test = None
        
        ## Network behaviour: data reduction vs data augmentation
        self.encoding_dim = encoding_dim
        self.denoising = denoising
        self.dropoutProba = dropoutProba
        
        ## Network technical specs
        self.optimizer = optimizer
        self.loss = loss
        self.nb_epoch = nb_epoch
        self.batch_size = batch_size
        self.activFirstLayer = activFirstLayer
        self.activSecondLayer = activSecondLayer
        
        ## Network features
        self.autoencoder = None
        self.history = None
        
        ## Network results
        self.score = None
        self.predictionTrain = None
        self.predictionTest = None
        self.residualTrain = None
        self.residualTest = None
        
        self.normalTests = None
        self.smoothnessGlobal = -1
        self.smoothnessTail = -1
        ##
        self.parametersSet = self.getParametersSet()
        
    # Inner neural network methods
    def buildAutoencoder(self,
                         inputDim):
        '''
        Network definition layer by layer
        '''
        # Input placeholder
        input_data = Input(shape=(inputDim,))

        # "encoded" is the encoded representation of the input
        encoded = Dense(self.encoding_dim,
                        activation=self.activFirstLayer,
                        activity_regularizer=regularizers.activity_l1(10e-5))(input_data)

        # "decoded" is the lossy reconstruction of the input
        decoded = Dense(inputDim, activation=self.activSecondLayer)(encoded)

        # This model maps an input to its reconstruction
        self.autoencoder = Model(input=input_data, output=decoded)
        self.autoencoder.compile(optimizer=self.optimizer,
                                 loss=self.loss)
        return True
    
    ## Fitting process
    def fitAutoencoder(self,
                       validation, # Dissociate train in x y to account for denoising eventuality
                       verbose=0):
        self.history = self.autoencoder.fit(self.x_train.as_matrix(), # Noisy input or not
                                            self.y_train.as_matrix(), # Clear output in any case
                                            nb_epoch=self.nb_epoch,
                                            batch_size=self.batch_size,
                                            shuffle=False,
                                            verbose=verbose,
                                            validation_data=(validation.as_matrix(),
                                                             validation.as_matrix()))
        return True
    
    ## Test process
    def predict(self,
                inputToPredict):
        return pd.DataFrame(self.autoencoder.predict(inputToPredict.as_matrix()),
                            index=inputToPredict.index.values,
                            columns=inputToPredict.columns.values)
    
    
    # Global process
    ## Full process: ML and metrics tracking
    def learningProcess(self, bplotPreds=True, bplotError=True, bplotResiduals=True):
        self.runML()
        self.setPredsAndResiduals()
        self.getMetrics()
        self.saveOutput()
        self.displayMetrics(bplotPreds=bplotPreds,
                            bplotError=bplotError,
                            bplotResiduals=bplotResiduals)
        self.releaseResources()
        return True
    


    
    ## ML process
    def runML(self, stocksFile='../donnees/clean/RET_PX_LAST.csv'):
        # Neural network inputs
        inputNN = ut.getInputs(scalingFactor=self.scalingFactor,
                               stocksFile=stocksFile)
        self.stockNames = inputNN.columns.values
        train, validation, self.test = ut.splitData(inputNN,
                                                    self.trainUB,
                                                    self.validationLB,
                                                    self.validationUB,
                                                    self.testLB)
        
        # Deal with denoising
        self.y_train = train
        if self.denoising: self.x_train = ut.noiseTrain2(train=train,
                                                         p=self.dropoutProba)
        else: self.x_train = train
        self.buildAutoencoder(inputDim=inputNN.shape[1])
        self.fitAutoencoder(validation=validation)
        return True

    # Utils
    ## Plot and log purpose
    def displayMetrics(self,
                       outputDir='../results/dae/neuralNetwork/graphes/',
                       bplotPreds=True,
                       bplotError=True,
                       bplotResiduals=True):
        logger.info('%s;%f;%f', '_'.join(self.parametersSet), self.score, self.smoothness)
        if bplotPreds: ut.plotPreds(self.predictionTest,
                                   self.test,
                                   outputDir=outputDir,
                                   parametersSet=self.parametersSet)
        if bplotError: ut.plotError(self.history,
                                   outputDir=outputDir,
                                   parametersSet=self.parametersSet)
        if bplotResiduals: ut.plotResiduals(residuals=self.residualTrain,
                                           outputDir=outputDir,
                                           parametersSet=self.parametersSet,
                                           who='train')
        if bplotResiduals: ut.plotResiduals(residuals=self.residualTest,
                                           outputDir=outputDir,
                                           parametersSet=self.parametersSet,
                                           who='test')
        return True
    
    ## Score purpose
    def getMetrics(self):
        # Score: loss on test set
        self.score = self.autoencoder.evaluate(self.test.as_matrix(),
                                               self.test.as_matrix(),
                                               verbose=0)
        # Score: smoothness on test set
        self.smoothness = measureSmoothness(dataFrame=self.residualTest, nMax=300, normThreshold=0.005)
        return True
    
    def getParametersSet(self, strForm=True):
        parametersSet = [self.scalingFactor, self.encoding_dim, self.denoising, self.dropoutProba, self.optimizer,
                         self.loss, self.nb_epoch, self.batch_size, self.activFirstLayer, self.activSecondLayer]
        if strForm: parametersSet = [str(parameter) for parameter in parametersSet]
        return parametersSet 
    
    def releaseResources(self):
        self.predictionTrain = None
        self.predictionTest = None
        self.residualTrain = None
        self.residualTest = None
        self.x_train = None
        self.y_train = None
        self.test = None
        return True
    
    def saveOutput(self, outputDir='../results/dae/neuralNetwork/predictions/'):
        rescaledTest = self.predictionTest / self.scalingFactor
        rescaledTest.to_csv(path_or_buf=outputDir + 'test_' + '_'.join(self.parametersSet) + '.csv',
                            sep=';',
                            index_label='Date')
        rescaledTrain = self.predictionTrain / self.scalingFactor
        rescaledTrain.to_csv(path_or_buf=outputDir + 'train_' + '_'.join(self.parametersSet) + '.csv',
                             sep=';',
                             index_label='Date')        
        
        return True
    
    def setPredsAndResiduals(self):
        # Predicitions on test and train
        # Train predictions are only useful from the finance point of view,
        # but strictly irrelevant from the machine learning point of view
        self.predictionTrain = self.predict(inputToPredict=self.y_train) # Clear input
        self.predictionTest = self.predict(inputToPredict=self.test)
        # Residuals
        self.residualTrain = self.y_train - self.predictionTrain
        self.residualTest = self.test - self.predictionTest
        return True


# In[ ]:

# Split denoising/non denoising ae in a fashion and easy way
def processFitting(params, denoising):
    encoding_dim, nb_epoch, batch_size, dropoutProba = (None, None, None, None)
    if not denoising: encoding_dim, nb_epoch, batch_size = params
    else: encoding_dim, nb_epoch, batch_size, dropoutProba = params
    autoencoder = BasicAutoencoder(scalingFactor=scalingFactor,
                                   encoding_dim=encoding_dim,
                                   denoising=denoising,
                                   dropoutProba=dropoutProba,
                                   optimizer=optimizer,
                                   loss=loss,
                                   nb_epoch=nb_epoch,
                                   batch_size=batch_size,
                                   activFirstLayer=activFirstLayer,
                                   activSecondLayer=activSecondLayer,
                                   trainUB=trainUB,
                                   validationLB=validationLB,
                                   validationUB=validationUB,
                                   testLB=testLB)
    autoencoder.learningProcess(bplotPreds=False, bplotError=True, bplotResiduals=False)
    return True

