#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 14:47:43 2018

@author: Jonas
"""

import import_SPP as sp
import import_forecast as fc
from Radiation_model import RadiationModel
import re
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import jit

class ARX:
    def __init__(self, date, muni_list, PredictFrom='11hour', na=1, nb=1,
                 Lambda=0.995, PredictAhead='60min', TimeResolution='15min',
                 WinLen='15day', return_mode=False):
        """
        Structure to handle forcasting of the SSP with an ARX model.

        Input:
            date, TimeStamp : A date in the interval 21-01-2017 to 31-12-2017
            nr, integer : The municipality number
            PredictionFrom, string : The hour which you want to predict from
            (the default is 11 am)
            na, integer : The number of autoregressive terms (the default is 1)
            nb, integer : The number of exogenous terms (the default is 1)
            PredictAhead, string : The number of minutes you want to predict
            ahead (the default is '60min')
            TimeResolution, string : The time resolution of the output
            (the default is '15min')
        """
        self.return_mode = return_mode
        self.Lambda = Lambda
        self.start = date
        self.first_data_date = date - pd.Timedelta(WinLen)
        self.end = date + pd.Timedelta(WinLen)
        self.PredictFrom = self.start + pd.Timedelta(PredictFrom)
        if muni_list == 'all':
            self.SPP = sp.import_SPP(self.first_data_date,self.end,
                                 muni_list = 'all').SPP
            self.muni_list = list(self.SPP.columns)
        else:
            self.muni_list = sorted(muni_list)
        self.na = na
        self.nb = nb
        self.quarter = datetime.timedelta(seconds=datetime.
                                          timedelta(minutes=15).total_seconds())
        self.PredictAhead = PredictAhead
        self.TimeResolution  = TimeResolution
        
        myfc = fc.import_muni_forecast(self.first_data_date, self.end,
                                       muni_list = self.muni_list)
        
        myfc_obj_simu = fc.import_muni_forecast_simu(self.first_data_date,
                                                     self.end,
                                                     muni_list=self.muni_list)
        
        myfc_simu = myfc_obj_simu(self.PredictFrom)
        

        

        self.RadModel_fc = RadiationModel(myfc)().resample('15min').interpolate(method='time')
        self.RadModel_simu = RadiationModel(myfc_simu)()
        
        
        self.index_set = self.RadModel_fc.index.intersection(self.RadModel_simu.index)
        self.RadModel_fc.loc[self.index_set] = self.RadModel_simu.loc[self.index_set]
        
        self.RadModel = self.RadModel_fc
        
        self.SPP = sp.import_SPP(self.first_data_date,self.end,
                                 muni_list = self.muni_list).SPP
                                
        self.SPP_res = pd.DataFrame(self.SPP.values[:len(self.RadModel.values)]
                                    - self.RadModel.values,\
                  index  = self.SPP.index[:-3],columns = self.SPP.columns)
        self.PredictNrQuaters = int(int(re.findall(r'\d+',
                                                   self.PredictAhead)[0])/15)

#    @profile
    @jit
    def __call__(self):
        """
        Uses call function to calculate the forecast
        """

        Y = self.SPP_res[:self.PredictFrom]
        

        u = self.RadModel

        idxset = self.SPP[:self.PredictFrom+
                          self.quarter*self.PredictNrQuaters].index
        test = pd.DataFrame(np.zeros((self.PredictNrQuaters,len(self.muni_list)))\
                            ,columns=self.muni_list)
        Y_forecast = Y.append(test, ignore_index=True).set_index(idxset)

        interval = pd.date_range(self.PredictFrom+self.quarter, idxset[-1],
                                 freq=self.TimeResolution)
        ParaDict = {}
        for muni_list_idx in self.muni_list:
            for t in interval:
                y_1, phi = self.split(Y_forecast[muni_list_idx].loc[:t-self.quarter],\
                                      u[muni_list_idx])
                parameters = self.para_esti(phi, y_1)
                Y_forecast[muni_list_idx].loc[t] = self.forecast(parameters,\
                          Y_forecast[muni_list_idx], u[muni_list_idx], t,)
                
                ParaDict[str(muni_list_idx)+','+str(t)] = {'Muni':muni_list_idx, \
                     'Parameters' : parameters}
            Y_forecast[muni_list_idx] = self.transform(Y_forecast, muni_list_idx)

        if self.return_mode == False:
            return self.zeros(Y_forecast[self.PredictFrom:])
        elif self.return_mode == True:
            return self.zeros(Y_forecast[self.PredictFrom:]), ParaDict

    def transform(self, Y, muni_nr):
        idx = Y.index
        return Y[muni_nr].loc[idx].values + self.RadModel[muni_nr].loc[idx].values
    
    @jit
    def split(self, ar, ar1):
        """
        Splits the data into the regressor vector (phi) and the data vector (y)
        """
        interval = ar.index

        Y = ar
        u = ar1
        phi1 = np.zeros((len(interval)-(self.na+self.nb),self.na+self.nb))

        y1 = np.zeros(len(interval)-(self.na+self.nb))
        j = 0
        for i in interval[self.na+self.nb:-1]:
            for q in range(self.na):
                phi1[j,q] = np.array(Y.loc[i-self.quarter*q])
            for k in range(self.nb):
                phi1[j,self.na+k] = u.loc[i-self.quarter*(k)]
            y1[j] =  Y.loc[i+self.quarter]
            j+=1

        return y1, phi1
    
    

#    @profile
    @jit
    def para_esti(self, ar, ar1):
        """
        Estimates the parameters through the normal equations with exponential
        forgetting
        """
        
        N = len(ar1)
        temp = np.zeros((self.na+self.nb, self.na+self.nb))
        temp1 = np.zeros(self.na+self.nb)
        for t in range(1,N):
            temp += self.Lambda**(N-t)*np.dot(ar[t].reshape(self.na+self.nb, 1),
                             ar[t].reshape(self.na+self.nb, 1).T)
            temp1 += self.Lambda**(N-t)*ar[t].T*ar1[t]
        return np.linalg.inv(temp).dot(temp1)

#    @profile
    @jit
    def forecast(self, parameters, X, u, t):
        """
        Calculates the forecast
        """
        a = parameters[0:self.na]
        b = parameters[self.na:self.na+self.nb]
        temp = 0
        for j in range(self.na):
            temp += a[j]*np.array(X.loc[t-self.quarter*(j+1)])
        for i in range(self.nb):
            temp += b[i]*np.array(u.loc[t-self.quarter*i])
        return temp

#    @profile
    def zeros(self, df):
        """
        If the output of the forecast is negative, it will be nulled.
        """
        n_df = df.where(df>0,0)
        return n_df
