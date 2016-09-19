
# coding: utf-8

# # Results analysis: utils
# A bunch of functions to plot many metrics

# In[ ]:

from bokeh.charts import Bar
from bokeh.io import output_notebook, show, output_file, reset_output, gridplot, save
from bokeh.plotting import figure
from bokeh.models import HoverTool, ColumnDataSource, Range1d, LinearAxis
from bokeh.palettes import YlOrRd7, Spectral7, RdBu10, PuBu9, RdYlGn10, OrRd9, YlGn9
from bokeh.models import Span

import numpy as np


# In[ ]:

TOOLS = "box_zoom, hover, pan, reset, resize, save, wheel_zoom"


# In[ ]:

def rawSort(results, metricToSortBy, metricsToKeep=None):
    '''
    Sort frame according to some variable
    '''
    if metricsToKeep != None: return results[metrics].sort_values(by=metricToSortBy).reset_index(drop=True)
    else: return results.sort_values(by=metricToSortBy).reset_index(drop=False)
    return False


# In[ ]:

def displayMetricEvolution(data,
                           metric,
                           outputDir='../results/dae/neuralNetwork/resultsAnalysis/',
                           configuration='concatParams'):
    reset_output()
    p = figure(tools=TOOLS, plot_width=350, plot_height=350)
    
    p.yaxis.axis_label = metric
    p.xaxis.axis_label = 'models'
    p.xaxis.axis_label_text_font_size = "12pt"
    p.yaxis.axis_label_text_font_size = "12pt"
    
    source = ColumnDataSource({metric: data[metric],
                               'index': data.index.values,
                               'conf': data[configuration]})
    
    p.line('index', metric, source=source)
    
    p.select_one(HoverTool).tooltips = [('conf','@conf')]
    
    output_notebook()
    show(p)
    output_file(outputDir + metric + 'Evolution.html')
    save(p)
    return True


# In[ ]:

def xyComparison(xdata,
                 ydata,
                 xlabel='Smoothness',
                 ylabel='Loss',
                 what='lossSmoothness',
                 outputDir='../results/dae/neuralNetwork/resultsAnalysis/'):
    
    N = len(xdata.values)
    counts = np.zeros((N, N))
    for value in xdata.values:
        i = ydata.values.tolist().index(value) # i for line
        j = xdata.values.tolist().index(value) # j for column
        counts[i,j] = 1

    xname = []
    yname = []
    color = []
    alpha = []
    for i, valY in enumerate(ydata):
        for j, valX in enumerate(xdata):
            xname.append(valX)
            yname.append(valY)
            if counts[i,j] == 1: color.append('red')
            else: color.append('white')
            alpha.append(1)

    source = ColumnDataSource(data=dict(xname=xname,
                                        yname=yname,
                                        colors=color,
                                        alphas=alpha,
                                        count=counts.flatten()))

    p = figure(x_axis_location="above",
               x_range=list(xdata),
               y_range=list(reversed(ydata.tolist())),
               tools=TOOLS)

    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.visible = None

    p.rect('xname',
           'yname',
           0.9,
           0.9,
           source=source,
           color='colors',
           line_color=None)

    p.select_one(HoverTool).tooltips = [('conf', '@xname')]   
    
    reset_output()
    output_notebook()
    show(p)
    
    output_file(outputDir + what + 'MatrixComparison.html')
    save(p)
    return True


# In[ ]:

def displayParameterEvolution(groupedData,
                              metrics,
                              parameter,
                              configuration='concatParams',
                              outputDir='../results/dae/neuralNetwork/resultsAnalysis/'):
    reset_output()
    output_notebook()
    
    grid = []
    subGrid = []
    for metric in metrics:
        p = figure(tools=TOOLS)
        p.xaxis.axis_label = parameter
        p.yaxis.axis_label = metric

        for name, group in groupedData:
            tmp = group.sort(columns=parameter)
            source = ColumnDataSource({parameter: tmp[parameter],
                                       metric: tmp[metric],
                                       'conf': tmp[configuration]})
            p.line(parameter, metric, source=source)
            
        p.select_one(HoverTool).tooltips = [('conf','@conf')]
        
        subGrid.append(p)
        
    grid.append(subGrid)
    p = gridplot(grid)
    
    show(p)
    output_file(outputDir + parameter + '.html')
    save(p)
    return True


# In[ ]:

def plotCombinedPortfoliosPerf(stratValues,
                               conf,
                               kind='gains', # Or returns
                               outputDir='../results/dae/portfolios/resultsAnalysis/',
                               benchmark=None): # Portfolios of portfolios actually
    reset_output()
    output_notebook()
    
    numlines = stratValues.shape[1]
    colors_list = list(reversed(RdBu10[0:numlines]))
    
    p = figure(x_axis_type="datetime")
    p.yaxis.axis_label = kind
    
    source = ColumnDataSource(stratValues)
    legs = stratValues.columns.values.tolist()
    
    for (colr, leg) in zip(colors_list, legs):        
        p.line(x=stratValues.index.values, y=leg, source=source, color=colr, legend=leg)
    
    hline = Span(location=0, dimension='width', line_color='black', line_width=2)
    p.renderers.extend([hline])
    
    p.legend.orientation = "top_left"
    
    if benchmark != None: p.line(benchmark.gains.index.values,
                                 benchmark.gains['cumulReturn'],
                                 color='darkgreen',
                                 legend='benchmark')
    
    show(p)
    output_file(outputDir + conf + '_' + kind +  '.html')
    save(p)
    return True


# In[ ]:

def getMetricsAction(portfolio):
    count = portfolio.count() / portfolio.shape[0]
    
    minn = portfolio.min()
    maxx = portfolio.max()
    
    sellPerc = portfolio[portfolio < 0].count() / portfolio.count() * 100
    buyyPerc = 100 - sellPerc
    
    name = portfolio.columns.values
    
    return count, minn, maxx, sellPerc, buyyPerc, name


# In[ ]:

def tradingRepartitionAction(portfolio, conf):
    reset_output()
    output_notebook()
    
    count, minn, maxx, sellPerc, buyyPerc, name = getMetricsAction(portfolio)
    
    colorRange = list(reversed(PuBu9))
    
    colors = []
    for stock in portfolio.columns.values:
        if count[stock] < 0.01: colors.append(0)
        elif count[stock] >= 0.01 and count[stock] <= 0.4: colors.append(1)
        elif count[stock] > 0.4 and count[stock] <= 0.5: colors.append(2)
        elif count[stock] > 0.5 and count[stock] <= 0.6: colors.append(3)
        elif count[stock] > 0.6 and count[stock] <= 0.7: colors.append(4)
        elif count[stock] > 0.7 and count[stock] <= 0.8: colors.append(5)
        elif count[stock] > 0.8 and count[stock] <= 0.9: colors.append(6)
        elif count[stock] > 0.9 and count[stock] <= 0.95: colors.append(7)
        elif count[stock] > 0.95: colors.append(8)
            

    xRange = []
    yRange = []
    for line in range(int(np.floor(portfolio.shape[1] / 10) + 1)):
        for column in range(10):
            xRange.append(str(column))
            yRange.append(str(line))

    source = ColumnDataSource(
        data=dict(
            xGroup = xRange,
            yGroup = yRange,
            name = name,
            count = count,
            minn = minn,
            maxx = maxx,
            sellPerc = sellPerc,
            buyyPerc = buyyPerc,
            type_color = [colorRange[colors[i]] for i in range(portfolio.shape[1])]
        )
    )
                      
    p = figure(title="Trading repartition by action",
               tools=TOOLS,
               x_range=[str(x) for x in range(10)],
               y_range=list(reversed([str(x) for x in range(int(np.floor(portfolio.shape[1] / 10) + 1))])))
    p.outline_line_color = None
    p.plot_width = 1200
    p.rect('xGroup', 'yGroup', 0.9, 0.9, source=source,
           fill_alpha=0.6, color="type_color")

    text_props = {
        "source": source,
        "angle": 0,
        "color": "black",
        "text_align": "center",
        "text_baseline": "middle"
    }

    p.text(x='xGroup', y='yGroup', text='name', text_font_size="9pt", **text_props)
    p.axis.visible = None
    p.grid.grid_line_color = None

    p.select_one(HoverTool).tooltips = [
        ("name", "@name"),
        ("participation", "@count"),
        ("min weight", "@minn"),
        ("max weight", "@maxx"),
        ("sellPerc", "@sellPerc"),
        ("buyyPerc", "@buyyPerc")
    ]

    show(p)
    output_file('../results/dae/portfolios/resultsAnalysis/tradingRepartitionAction_'+ conf + '.html')
    save(p)
    return True


# In[ ]:

def getMetricsDay(portfolio, absGain):
    count = portfolio.count(axis=1) / portfolio.shape[1] * 100
    
    sell = portfolio[portfolio < 0].count(axis=1)
    buyy = portfolio[portfolio > 0].count(axis=1)
    
    days = absGain.index.strftime('%Y-%m-%d')
    
    return count, sell, buyy, days


# In[ ]:

def tradingRepartitionDay(portfolio, absGain, conf):
    reset_output()
    output_file('../results/dae/portfolios/resultsAnalysis/tradingRepartitionDay_'+ conf + '.html')
    
    count, sell, buyy, days = getMetricsDay(portfolio, absGain)
    
    colorRange = OrRd9[0:5] + list(reversed(YlGn9[0:5]))
    
    colors = []
    for day in absGain.index.values:
        if absGain[day] < -5000: colors.append(0)
        elif absGain[day] >= -5000 and absGain[day] <= -2500: colors.append(1)
        elif absGain[day] > -2500 and absGain[day] <= -1000: colors.append(2)
        elif absGain[day] > -1000 and absGain[day] <= -500: colors.append(3)
        elif absGain[day] > -500 and absGain[day] < 0: colors.append(4)
        elif absGain[day] == 0: colors.append(-1)
        elif absGain[day] > 0 and absGain[day] <= 500: colors.append(5)
        elif absGain[day] > 500 and absGain[day] <= 1000: colors.append(6)
        elif absGain[day] > 1000 and absGain[day] <= 2500: colors.append(7)
        elif absGain[day] > 2500 and absGain[day] <= 5000: colors.append(8)
        elif absGain[day] > 5000: colors.append(9)

    xRange = []
    yRange = []
    for line in range(int(np.floor(absGain.shape[0] / 20) + 1)):
        for column in range(20):
            xRange.append(str(column))
            yRange.append(str(line))

            
    type_color = []
    for i, color in enumerate(colors):
        if color != -1: type_color.append(colorRange[colors[i]])
        else: type_color.append('grey')
    
    source = ColumnDataSource(
        data=dict(
            xGroup = xRange,
            yGroup = yRange,
            absGain = absGain,
            days = days,
            count = count,
            sell = sell,
            buyy = buyy,
            type_color = type_color
        )
    )
                      
    p = figure(title="Trading repartition by day",
               tools=TOOLS,
               x_range=[str(x) for x in range(20)],
               y_range=list(reversed([str(x) for x in range(int(np.floor(portfolio.shape[0] / 20) + 1))])))
    p.outline_line_color = None
    p.plot_width = 1800
    p.plot_height = 1800
    p.rect('xGroup', 'yGroup', 0.9, 0.9, source=source,
           fill_alpha=0.6, color="type_color")

    text_props = {
        "source": source,
        "angle": 0,
        "color": "black",
        "text_align": "center",
        "text_baseline": "middle"
    }

    p.text(x='xGroup', y='yGroup', text='days', text_font_size="9pt", **text_props)
    p.axis.visible = None
    p.grid.grid_line_color = None

    p.select_one(HoverTool).tooltips = [
        ("day", "@days"),
        ("% actions involved", "@count"),
        ("sold", "@sell"),
        ("bought", "@buyy"),
        ("absGain", "@absGain")
    ]

    save(p)
    show(p)
    return True

