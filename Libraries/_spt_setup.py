# -*- coding: utf-8 -*-
"""
Created on Wed May  2 14:42:21 2018

@author: Tobias
"""

import import_SPP as SPP
import pandas as pd
import import_forecast as fc
import Radiation_model as rad
import WS_WD_mod
import numpy as np
from datetime import time as d_time

def _setup(t_start,t_end,muni_list,hours = 'all'):
    """
    Internal function but can be used externally with specified arguments. 
    Imports relevant spp data and forecast data. Then it removes the diurnal
    part by using the radiation model and models for the forecast. Finally
    returns everyting as different classes. This function is used for setting
    up the fitting of the spatio-temporal correction model. Is also used to
    setup for forcasting with the correction model.
    """
    if hours == 'all':
        hours = ['00:00','23:45']
        
    M = len(muni_list)
    
    if t_start.time() != d_time(0,0) or t_end.time() != d_time(0,0):
        raise(ValueError("t_start and t_end should be whole dates only, i.e \
                     hours = 0 and minutes = 0. \nUse the hours argument to\
                     get less hours on a day")) 
    
    #import the information
    SPP_scada = SPP.import_SPP(t_start, t_end,\
                         hours = hours, muni_list = muni_list).SPP
    
    #replace minutes to zero for the fc forecast
    hours_fc = ["%s:00"%(hours[0][:2]),"%s:00"%(hours[1][:2])] 
    
    #There might miss 3 samples in the end of the day so we and another day
    #to combat that. 
    if pd.Timestamp(hours[1]).time() > d_time(23,0): 
        t_end_fc = t_end + pd.Timedelta(days = 1)
    else:
        t_end_fc = t_end
        
    fc_rng = fc.import_muni_forecast(t_start,t_end_fc, muni_list = muni_list,\
                                     hours = hours_fc,\
                                     info = 'all')
    
    fc_rng.hours = hours
    minutes = (eval(hours[0][-2:]),eval(hours[1][-2:]))
    
    #interpolate wind speed and direction
    fc_rng.WS = fc_rng.WS.resample('15min').interpolate(medthod = "time")
    fc_rng.WD = fc_rng.WD.resample('15min').interpolate(medthod = "time")
    
    # =========================================================================
    # remove diurnal patterns using different models
    # =========================================================================
    
    #for SPP
    rad_mod = rad.RadiationModel(fc_rng,minutes = minutes) 
    rad_vals = rad_mod()
    rad_vals = rad_vals.loc[SPP_scada.index] #remove the extra day if necessary
    SPP_res = pd.DataFrame(SPP_scada.values - rad_vals.values,\
                      index  = SPP_scada.index,\
                      columns = SPP_scada.columns)
    
    #for wind speed
    WS_mod = WS_WD_mod.WS_mod()
    rng = pd.date_range(t_start,t_end+pd.Timedelta(days = 1,hours = -1),\
                        freq = '15min')
    rng = rng[rng.indexer_between_time(hours_fc[0],hours_fc[1])]
    WS_mu = WS_mod(rng,muni_list)
    WS_res = pd.DataFrame(fc_rng.WS.loc[rng].values - WS_mu.values,\
                      index  = rng,columns = fc_rng.WS.columns)
   
    # for wind direction
    WD_mod = WS_WD_mod.WD_mod()
    WD_mu = WD_mod(muni_list)    
    WD_res = np.deg2rad(fc_rng.WD.loc[rng] - WD_mu.values.reshape(M,))
    return(SPP_scada,rad_vals,SPP_res,WS_res,WD_res)    

