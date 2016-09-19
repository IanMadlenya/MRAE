
# coding: utf-8

# # Utils

# In[1]:

from bokeh.palettes import Spectral7
from bokeh.plotting import figure, output_file, show, save, reset_output
from bokeh.io import gridplot, output_notebook, output_file

import numpy as np
import pandas as pd


# ## Input data frame setup

# ### Load data

# In[2]:

# Concatenate interest variables for each stock in a single data frame
# E.g.:
# stock1|stock2|...|stockn
# return_stock1_date1|return_stock2_date1|...|Return_stockn_date1
# return_stock1_date2|return_stock2_date2|...|Return_stockn_date2
def buildInput(stocks, feature):
    initKey = list(stocks.keys())[0]
    inputDataFrame = stocks[initKey][feature].to_frame(name=initKey)
    
    for i, stock in enumerate(stocks):
        if stock != initKey:
            inputDataFrame = pd.concat([inputDataFrame, stocks[stock][feature].to_frame(stock)], axis=1)
            
    return(inputDataFrame)


# In[3]:

def getInputs(scalingFactor,
              stocksFile):
    inputNN = pd.read_csv(filepath_or_buffer =stocksFile,
                          sep=';',
                          header=0,
                          index_col='Date',
                          parse_dates=True) * scalingFactor
    return(inputNN)


# In[4]:

def splitData(inputData, trainUB, validationLB, validationUB, testLB):
    inputData_train = inputData.loc[:trainUB]
    inputData_validation = inputData.loc[validationLB:validationUB]
    inputData_test = inputData.loc[testLB:]
    return(inputData_train, inputData_validation, inputData_test)


# ### Noise data
# 
# In case of denoising autoencoder

# In[5]:

def noiseTrain(train, sigma):
    res = train.copy()
    res += sigma * np.random.normal(loc=0.0, scale=1.0, size=res.shape)
    return(res)
def noiseTrain2(train, p=0.25):
    res = train.copy()
    tmp = np.random.uniform(0, 1, res.shape)
    tmp[tmp < p] = 0
    tmp[tmp > p] = 1
    
    return(np.multiply(res, tmp))


# ## Plot network results

# In[1]:

def plotPreds(prediction, test, outputDir, parametersSet):
    reset_output()
    stocks = test.columns.values
    
    dataTest = test.reset_index()
    output_file(outputDir + '_'.join(parametersSet) + '_predPerf.html')
    colors_list = ['green', 'red']
    
    grid = []
    subGrid = []
    for i, stock in enumerate(sorted(stocks)):
        if i % 3 == 0 and i != 0:
            grid.append(subGrid)
            subGrid = []
        legends_list = [stock, 'reconstruction']
        xs = [dataTest['Date'], dataTest['Date']]
        ys = [dataTest[stock], prediction[stock]]
        
        p = figure(x_axis_type="datetime",
                   y_axis_label = "Log-return")
        for (colr, leg, x, y ) in zip(colors_list, legends_list, xs, ys):
            p.line(x, y, color=colr, legend=leg)
        subGrid.append(p)
    p = gridplot(grid)
    save(p)
    return True

def plotError(history, outputDir, parametersSet):
    reset_output()
    output_file(outputDir + '_'.join(parametersSet) + '_loss.html')
    colors_list = ['green', 'red']
    p = figure(x_axis_label='iteration',
               y_axis_label='average loss',plot_width=350, plot_height=350)
    p.xaxis.axis_label_text_font_size = "12pt"
    p.yaxis.axis_label_text_font_size = "12pt"
    legends_list = ['validation loss', 'training loss']
    xs = [history.epoch, history.epoch]
    ys = [history.history['loss'], history.history['val_loss']]
    
    for (colr, leg, x, y ) in zip(colors_list, legends_list, xs, ys):
            p.line(x, y, color=colr, legend=leg)
    save(p)
    return True

def plotResiduals(residuals, outputDir, parametersSet, who):
    reset_output()
    stocks = residuals.columns.values
    
    res = residuals.reset_index()
    output_file(outputDir + '_'.join(parametersSet)  + '_residuals_' + who + '.html')
    
    grid = []
    subGrid = []
    for i, stock in enumerate(sorted(stocks)):
        if i % 3 == 0 and i != 0:
            grid.append(subGrid)
            subGrid = []
        p1 = figure(title=stock + ' ' + who + ' residuals', background_fill_color="#E8DDCB", x_axis_label='r - r_hat')
        p1.yaxis.visible = None
        p1.legend.location = "top_left"
        hist, edges = np.histogram(res[stock], density=True, bins=25)
        p1.quad(top=hist,
                bottom=0,
                left=edges[:-1],
                right=edges[1:],
                fill_color="#036564",
                line_color="#033649")
        subGrid.append(p1)
    p = gridplot(grid)
    save(p)
    return True

