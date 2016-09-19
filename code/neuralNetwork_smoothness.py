
# coding: utf-8

# # Smoothness
# 
# Implement residual distribution smoothness:
# * Compute gradient of the full, right and left tail distributions (left and right tails are arbitrary quantiles)
# * Compute gardient norm and produce a score
# * Average scores among distributions

# In[ ]:

import numpy as np


# ## Smoothness score
# 
# For a single stock

# In[ ]:

def smoothnessDensityMeasure(array, nMax, normThreshold, quantile):
    '''
    Return smoothness score for the full, right and left tail distributions
    Score is based on the number of derivatives needed to get the gradient norm close to 0
    The smaller the score, the smoothest the distribution
    
    nMax: maximal number of derivations
    normThreshold: threshold under which something is considered close enough to 0
    quantile: which quantile is considered to fit distribution tails
    '''
    distribution = sorted(array, reverse=False) # Order residuals
    
    # Distribution is tail-cut
    ## Find quantiles
    lBound = np.percentile(distribution, quantile)
    rBound = np.percentile(distribution, 100 - quantile)
    
    ## Values associated to quantiles
    lDistrib = [val for val in distribution if val <= lBound]
    rDistrib = [val for val in distribution if val >= rBound]
    lBoundIndex = len(lDistrib)
    rBoundIndex = len(distribution) - len(rDistrib)
    
    # Number of derivatives needed to get gradient norm close to 0; nMax by default
    fullSmoothness = nMax
    lSmoothness = nMax
    rSmoothness = nMax
    
    # Boolean temporary variables to track which parts of the distribution still need to be considered
    full = False # Full distribution
    l = False # Left distribution
    r = False # Right distribution
    
    for n in range(1, nMax + 1):
        distribution = np.gradient(distribution)        
        if not full and np.linalg.norm(distribution) <= normThreshold:
            full = True
            fullSmoothness = n
        if not l and np.linalg.norm(distribution[0:lBoundIndex]) <= normThreshold / 2:
            l = True
            lSmoothness = n
        if not r and np.linalg.norm(distribution[rBoundIndex:len(distribution)]) <= normThreshold / 2:
            r = True
            rSmoothness = n
    return (fullSmoothness, [lSmoothness, rSmoothness])


# ## Smoothness global process
# 
# For all stocks

# In[ ]:

def measureSmoothness(dataFrame, nMax, normThreshold, quantile=0.1):
    fullSmoothness = []
    tailSmoothness = []
    for stock in dataFrame.columns.values:
        fSmoothness, tSmoothness = smoothnessDensityMeasure(dataFrame[stock], nMax, normThreshold, quantile)
        fullSmoothness.append(fSmoothness)
        tailSmoothness.append(np.mean(tSmoothness))
    return 1./3 * np.mean(fullSmoothness) + 2./3 * np.mean(tailSmoothness)

