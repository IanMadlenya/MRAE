
# coding: utf-8

# In[ ]:

from neuralNetwork import *


# In[ ]:

if __name__ == "__main__":
    # Log NN performances
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # create a file handler
    handler = logging.FileHandler('../results/dae/neuralNetwork/performances_ML.csv', mode='w')
    handler.setLevel(logging.INFO)
    # create a logging format
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(handler)
    logger.info('concatParams;loss;smoothness')
    #####
    
    # Neural Network parameters
    ## Scaling
    scalingFactor=10

    ## Data split
    trainUB = '2012-12'
    validationLB = '2013-1'
    validationUB = '2013-12'
    testLB = '2014'

    ## Autoencoder parameters
    comprDims = [25, 50, 75, 100]
    augmeDims = [170, 190, 210]
    dropoutProbas = [0.3, 0.4]
    optimizer = 'adadelta'
    loss = 'mse'
    activFirstLayer = 'tanh'
    activSecondLayer = 'linear'

    ## Training parameters
    nb_epochs = [50, 100, 150, 200, 300]
    batch_sizes = [1, 10]

    paramListCompr = [comprDims, nb_epochs, batch_sizes]
    paramListAugme = [augmeDims, nb_epochs, batch_sizes, dropoutProbas]
    #####
    
    # main()
    for params in itertools.product(*paramListCompr): processFitting(params=params, denoising=False)
    for params in itertools.product(*paramListAugme): processFitting(params=params, denoising=True)

