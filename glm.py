# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 13:37:57 2018

@author: Tobias
"""

import pandas as pd
import numpy as np
from scipy.stats import t

import statsmodels.api as sm
import statsmodels.formula.api as smf

class lm:
    def __init__(self,X,y):
        self.X = np.array(X)
        self.y = y
        self.ybar = np.mean(y)
        self.beta = np.linalg.lstsq(X,y)[0]
        self.yhat = X.dot(self.beta)
        self.N,self.k = np.shape(X)
        self.D = np.linalg.norm(self.y - self.yhat)**2       
    
    def t_test(self):
        t = np.zeros(self.k)
        for j in range(self.k): #get t-values for each parameter
            t[j] = self.beta[j]/(np.sqrt(self.var*self.X_semi_projected[j,j]))
        return(t)
    
    def t_sig(self):
        t_sig = np.zeros(self.k)
        t_dist = t(self.N - self.k)
        for j in range(self.k):
            t_sig[j] = 2*(1 - t_dist.cdf(abs(self.t_values[j])))
        return(t_sig)
        
    def install_R2(self):
        y_null = np.linalg.norm(self.y - self.ybar)**2
        self.R2 =  1 - self.D/y_null
        self.R2adj = 1 - ((self.N-1)/(self.N - self.k))*(1 - self.R2)

    def install_t_values(self):
        self.var = self.D/(self.N - self.k)
        self.X_semi_projected = np.linalg.inv((self.X.T).dot(self.X))
        self.t_values = self.t_test()
        self.t_sig = self.t_sig() #p-values
    
class bck_eli_glm:
    def __init__(self,X,y,alpha,mode = "t-test"):
        self.X = pd.DataFrame(X)
        self.y = y
        self.a = alpha
        if mode == "t-test":
            res =  self.perform_bck_eli_ttest()
        elif mode == "adj_R2":
            res = self.perform_bck_eli_adj_r2()
        self.mod_eli, self.var_nr, self.rm_hist, self.p_hist = res
        
    def perform_bck_eli_ttest(self):
        X_temp = self.X.copy()
        H = lm(self.X,self.y)
        H.install_t_values()
        k = H.k
        rm_hist = []
        p_hist = {'H00': H.t_sig}
        i = 0
        while True in np.greater(H.t_sig, self.a) and H.k > 1:
            print(i)
            rm = X_temp.columns[H.t_sig.argmax()]  #Remove parameter with largest t_sig value
            rm_hist.append(rm)
            X_temp = X_temp.drop(rm,axis = 1)
            H = lm(X_temp,self.y)
            H.install_t_values()
            tsigcurrent = np.zeros(k)
            for nr in range(H.k):
                tsigcurrent[X_temp.columns[nr]] = H.t_sig[nr] 
            i += 1
            s = str(i)
            s = zeropad_hourstring(s)
            p_hist['H%s'%s] = tsigcurrent
        hist_df = pd.DataFrame(p_hist).transpose()
        hist_df.rename(columns={hist_df.columns[0]:"Current_model"},\
                                inplace = True)
#        hist_df.replace(0,np.nan,inplace = True) #Replace zeros with nan
        return H,X_temp.columns, rm_hist, hist_df
    
    def perform_bck_eli_adj_r2(self):
        X_temp = self.X.copy()
        H = lm(self.X,self.y)
        H.install_R2()
        k = H.k
        rm_hist = []
        R2_temp = H.R2adj
        R2_all_lower = self.get_all_R2(X_temp)
        R2_hist = {'H00': np.hstack([R2_temp,R2_all_lower])}
        i = 0
        while True in np.greater(R2_all_lower,R2_temp) and H.k > 1:
            print(i,R2_hist)
            #Remove parameter to get the largest R2 adjusted value
            rm = X_temp.columns[R2_all_lower.argmax()]
            rm_hist.append(rm)
            X_temp = X_temp.drop(rm,axis = 1) #Remove column from design matrix
            H = lm(X_temp,self.y)
            H.install_R2() #update R2 value for current model
            R2_temp = H.R2adj
            #Get values for removing parameter in each column
            R2_all_lower = self.get_all_R2(X_temp) 
            
            #Document all values
            R2_Hx = np.zeros(k)
            for nr in range(H.k):
                R2_Hx[X_temp.columns[nr]] = R2_all_lower[nr] 
            i += 1
            s = str(i)
            s = zeropad_hourstring(s)
            R2_hist['H%s'%s] = np.hstack([R2_temp,R2_Hx])
        
        hist_df = pd.DataFrame(R2_hist).transpose()
        hist_df.rename(columns={hist_df.columns[0]:"Current_model"},\
                                inplace = True)
        hist_df.replace(0,np.nan,inplace = True) #Replace zeros with nan
        return H,X_temp.columns, rm_hist, hist_df
    
    def get_all_R2(self,X):
        m = np.shape(X)[1] #nr of variables
        R2 = np.zeros(m)
        for i in range(m):
            X_temp = X.drop(X.columns[i],axis = 1)
            mod_temp = lm(X_temp,self.y)
            mod_temp.install_R2()
            R2[i] = mod_temp.R2adj
        return(R2)

def zeropad_hourstring(string):
    if len(string) == 1:
        return "0" + string
    else:
        return string
