    # -*- coding: utf-8 -*-
"""
Created on Sun May  6 16:12:22 2018

@author: Tobias
"""
import numpy as np


def BIC(y,yhat,k):
    """
    Calculates the basean information criterion given the the model is 
    normally distributed. 
    
    Parameters
    ----------
    
    y : numpy.ndarray
        model values
    yhat : numpy.ndarray
        estimated model
    k : int
        model order
        
    Returns
    -------
    BIC : float
        Bayesian information criterion for the model. 
        
    Examples
    --------
    >>> y = [3,3,4] #real value
    >>> yhat = [2,3,4] #estimate using two parameters
    >>> BIC(y,yhat,2)
    3      
    """
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    
    N = yhat.size
    y = y.reshape(N,)
    yhat = yhat.reshape(N,)
    residuals = y-yhat
    sse = np.sum(residuals**2) #sum of squared errors
    N = len(y)
    Lhat = sse/(N-k)
    return(N*np.log(Lhat) + k*np.log(N))
        
def BIC_edit(y,y_hat,k,vola_val = None,b=None):
    """
    Calculates the basean information criterion given the the model is 
    normally distributed. But the k parameter is altered to favor more 
    complicated models a bit more
    
    Parameters
    ----------
    
    y : numpy.ndarray
        model values
    yhat : numpy.ndarray
        estimated model
    k : int
        model order
    vola_val : numpy.ndarray
        Volatility value
    b : numpy.ndarray
        b coefficient
    
    Returns
    -------
    BIC : float
        Bayesian information criterion for the model. 
        
    Examples
    --------
    >>> y = [3,3,4] #real value
    >>> yhat = [2,3,4] #estimate using two parameters
    >>> BIC(y,yhat,2)
    3      
    """
    y = np.asarray(y)
    y_hat = np.asarray(y_hat)
    vola_val = np.asarray(vola_val)
    
    if y.size != y_hat.size or vola_val.size != y_hat.size:
        raise(ValueError("y and yhat and vola val should be of same size now\n\
            size(y) = %d and size(yhat) = %d"%(y.size,y_hat.size)))
    N = y_hat.size
    y = y.reshape(N,)
    y_hat = y_hat.reshape(N,)
    vola_val = vola_val.reshape(N,)
    try:
        Tinv = np.diag(1/(np.sqrt(b[0] + b[1]*vola_val))) #cholesky factorication
        #of dispersion matrix
    except:
        raise(ValueError("Vola val not given"))
    residuals = Tinv.dot(y - y_hat) #transformed data
    sse = np.sum(residuals**2) #sum of squared errors
    N = len(y)
    sighat = sse/(N-k)
    return(N*np.log(sighat) + k*np.log(N))

def AIC(y,yhat,k):
    """
    Calculates the basean information criterion given the the model is 
    normally distributed. 
    
    Parameters
    ----------
    
    y : numpy.ndarray
        model values
    yhat : numpy.ndarray
        estimated model
    k : int
        model order
        
    Returns
    -------
    BIC : float
        Bayesian information criterion for the model. 
        
    Examples
    --------
    >>> y = [3,3,4] #real value
    >>> yhat = [2,3,4] #estimate using two parameters
    >>> BIC(y,yhat,2)
    3      
    """
    residuals = y-yhat
    sse = np.sum(residuals**2) #sum of squared errors
    N = len(y)
    Lhat = sse/N
    return(2*k + N*np.log(Lhat))
    
def MSE(y,yhat):
    """
    Computes the MSE (mean squared error) for at model and it's estimation
    """
    #
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    if y.size != yhat.size:
        raise(ValueError("y and yhat should be of same size now\n\
            size(y) = %d and size(yhat) = %d"%(y.size,yhat.size)))
    N = yhat.size
    y = y.reshape(N,)
    yhat = yhat.reshape(N,)
    
    res = y - yhat
    sse = np.sum(res**2) #sum squared errors
    MSE = sse/N
    return(MSE)
    
def RMSE(y,yhat):
    """
    Computes the RMSE (Root mean squared error) of a model and it's estimation.
    """
    return(np.sqrt(MSE(y,yhat)))

def MAE(y,yhat):
    """
    Computes the MAE (Absolute mean error) of a model and it's estimation.
    """
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    if y.size != yhat.size:
        raise(ValueError("y and yhat should be of same size now\n\
            size(y) = %d and size(yhat) = %d"%(y.size,yhat.size)))
    N = yhat.size
    y = y.reshape(N,)
    yhat = yhat.reshape(N,)
    
    res = y - yhat
    se = np.sum(np.abs(res))
    MAE = se/N
    return(MAE)

def SMAPE(y,yhat):
    """
     Computes the SMAPE (Symmetric mean absolute percentage error) of a
     model and it's estimation.
    """
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    if y.size != yhat.size:
        raise(ValueError("y and yhat should be of same size now\n\
            size(y) = %d and size(yhat) = %d"%(y.size,yhat.size)))
    N = yhat.size
    y = y.reshape(N,)
    yhat = yhat.reshape(N,)
    
    res = y-yhat
    SMAPE = 2*np.mean((np.abs(res)/(np.abs(y)+np.abs(yhat))))
    return(SMAPE)

def R2(y,yhat):
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    if y.size != yhat.size:
        raise(ValueError("y and yhat should be of same size now\n\
            size(y) = %d and size(yhat) = %d"%(y.size,yhat.size)))
    N = yhat.size
    y = y.reshape(N,)
    yhat = yhat.reshape(N,)
    yavg = np.mean(y)
    
    denominator = np.linalg.norm(y - yhat)**2  
    y_null = np.linalg.norm(y - yavg)**2
    R2 =  1 - denominator/y_null
#    ybar = np.sum(y)/N       # or sum(y)/len(y)
#    ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
#    sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
#    R2 = ssreg / sstot
    return(R2)