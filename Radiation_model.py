#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 14:11:39 2018

@author: Jonas
"""

import numpy as np
import pandas as pd
import os
import sys
import datetime
import import_SPP as sp
import import_forecast as fc
from utilities import return_to_root
stem_path = 'Fortrolig_data/stem_data'
root = return_to_root()
os.chdir(root + stem_path)
stem_data = pd.read_excel('Kommune_GridNr.xlsx',header = 0).set_index('Kommune_Nr')
Kommune_List = list(pd.read_excel('Kommune_GridNr.xlsx',header = 0)['Kommune_Nr'])


class RadiationModel:
    def __init__(self,fc_obj,norm = False,TimeResolution = '15min',\
                 minutes = (0,0)):
        """
        RadiationModel() returns a DataFrame of which the size depends on the input,
        the time resolution for the output is standard a quarter resolution but can
        be changed the "TimeResolution" parameter
        
        Input: The input is a forecast object, see import_forecast and the
        municipality number assioate with the data
        minutes : tup/list
           controll beginning and end minutes for a day
        Output: The output is is the the model as Pandas DataFrame within the 
        Time period as the input
        """
        
        self.fc_obj = fc_obj
        root = return_to_root()
        coef_path = 'Scripts/libs_and_modules/coef/'
        os.chdir(root + stem_path)
        self.MList = np.array(pd.read_excel('Kommune_GridNr.xlsx',header = 0)['Kommune_Nr'])
        os.chdir(root + coef_path)
        self.beta = np.load('red_model_beta.npy')
        self.kept_beta_index = np.load('red_model_info.npy')[5:]-5 # -5 because the first 5 coefs aren't for municipalities
        self.GHI = fc_obj.GHI*10**(-3) # Skalers til MW
        self.KNr = fc_obj.muninr
        self.norm = norm
        self.TimeResolution = TimeResolution
        self.MuniNrCoef = self.MList[self.kept_beta_index]
        self.t_start = self.fc_obj.GHI.index[0].date()
        self.t_end = self.fc_obj.GHI.index[-1].date()
        self.min = minutes #used when the rad model for the entire day is desired
        #minute can also make not hourly intervals in the beginning of the day
        
        
        self.IndxSet = self.findIndx()
    
    def findIndx(self):
        return pd.DataFrame(list(range(5,len(self.kept_beta_index)+5)),self.MuniNrCoef)
   
    def get_muni_coef(self,j):
        if j in self.MList[self.kept_beta_index]:
            return self.beta[self.IndxSet[0][j]]
        else:
            return 0
    
    def get_season_coef(self,i):
        month = self.GHI.index[i].month
        if month in range(1,4):
            return self.beta[2]
        elif month in range(4,7):
            return self.beta[3]
        elif month in range(7,10):
            return self.beta[4]
        else:
            return self.beta[5]
        
    def Scale(self,model):
        instp_df = sp.import_instp(self.t_start,self.t_end,\
                                   muni_list=list(self.KNr))

        for j in self.KNr:
            for t in pd.date_range(self.t_start,self.t_end,freq = "D"):
                instP = instp_df[j].loc[t]
                model[j].loc[t:t+pd.Timedelta(hours = 23,minutes = 45)] *= instP
        return model
    def TimeScale(self,radmodel):
        radmodel = radmodel.resample(self.TimeResolution).\
        interpolate(method='time')
        if self.fc_obj.hours != 'all':
            rng = pd.date_range(pd.Timestamp(self.t_start) \
                                + pd.Timedelta(minutes =self.min[0]),\
                                pd.Timestamp(self.t_end) + \
                                pd.Timedelta(hours = 23,minutes = self.min[1])\
                                ,freq = self.TimeResolution)
            rng = rng[rng.indexer_between_time(self.fc_obj.hours[0],\
                                       self.fc_obj.hours[1])]
            radmodel = radmodel.loc[rng]
            #some of the values might be nan depending on minute aguments
            #they are removed here
            idx = np.where(np.isnan(radmodel[radmodel.keys()[0]]))[0]
            radmodel = radmodel.drop(radmodel.index[idx])
        return radmodel

    
    def __call__(self):
        length = len(self.GHI)
        model = self.fc_obj.GHI.copy()
        for j in self.KNr:
            for i in range(length):
                model[j][i] = (self.beta[0] + self.get_season_coef(i) +\
                     self.get_muni_coef(j))*self.GHI[j][i]
        if self.norm == True:
            return self.TimeScale(model)
        else:
            return self.Scale(self.TimeScale(model))
        


