# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 11:28:35 2018

@author: Tobias
"""

import numpy as np
from utilities import return_to_root
import pandas as pd

class WS_mod:
    """
    Handles normalisation of wind speed given municipality by fitted model.
    Can be called on pandas date_range to give values.
    
    Parameters
    ----------
    
    freq : str, pandas frequency type
       frequency for the model set to 1 hour as default
       
    Returns
    -------
    WS_norm: The Average wind speed given municipality and time
    
    Examples
    --------
    >>> t0 = pd.Timestamp(2017,5,1); t1 = pd.Timestamp(2017,5,5)
    >>> rng = pd.date_range(t0,t1,freq = "H")
    >>> mod = WS_mod()
    >>> WS_avg = mod(rng,[849,]) #Average for a certain municipality
    >>> WS_avg[:10]
                              849
    Time                         
    2017-05-01 00:00:00  8.032863
    2017-05-01 01:00:00  7.965725
    2017-05-01 02:00:00  7.855917
    2017-05-01 03:00:00  7.717551
    2017-05-01 04:00:00  7.572309
    2017-05-01 05:00:00  7.444681
    2017-05-01 06:00:00  7.356388
    2017-05-01 07:00:00  7.321410
    2017-05-01 08:00:00  7.342901
    2017-05-01 09:00:00  7.412769
    """
    def __init__(self,freq = "H"):
        root = return_to_root()
        coef_path = "Scripts/libs_and_modules/coef/WS_coef.p"
        self.coef_df = pd.read_pickle(root + coef_path)
        self.muni_list = self.coef_df.columns
        self.core_func_vec = np.vectorize(self.core_func)
        self.freq = freq
    
    def core_func(self, month, hour, muni): #Core function for calling
        coef = self.coef_df[muni]
        if 1 <= month <= 3:
            const = coef[0]
        elif 4 <= month <= 6:
            const = coef[1] 
        elif 7 <= month <= 9:
            const = coef[2]
        elif 10 <= month <= 12:
            const = coef[3]
        return(const + coef[4]*np.sin((2*np.pi*hour)/24) + \
                     + coef[5]*np.cos((2*np.pi*hour)/24) + \
                     + coef[6]*np.sin((4*np.pi*hour)/24) + \
                     + coef[7]*np.cos((4*np.pi*hour)/24))
    
    def __call__(self, t, muni_list='all'):
        h = np.array(t.hour + t.minute/60)
        months = np.array(t.month)
        WS_avg = {'Time': t}
        if muni_list == "all":
            muni_list = self.muni_list
        
        for muni in muni_list:
            WS_avg[muni] = self.core_func_vec(months, h, muni)
        return(pd.DataFrame(WS_avg).set_index("Time"))

class WD_mod:
    """
    Handles normalisation of wind direction given municipality by fitted model.
    Can be called on different muncipalities
    
    Parameters in call
    ------------------
    
    mode : str
       deg for degrees and rad for radians
       
    Returns
    -------
    WS_norm: The Average wind direction given municipality
    
    Examples
    --------
    >>> wd_avg = WD_mod()
    >>> wd_avg([849,851,860]) #In degrees
                        coef
    Municipality            
    849           247.835283
    851           246.921717
    860           244.941953
    >>> wd_avg([849,851,860],"rad") #In radians
                      coef
    Municipality          
    849           4.325542
    851           4.309597
    860           4.275044
    """
    def __init__(self):
        root = return_to_root()
        coef_path = "Scripts/libs_and_modules/coef/WD_coef.p"
        self.coef_df  = pd.read_pickle(root + coef_path)
        self.muni_list = self.coef_df.index
    
    def __call__(self,muni_list = 'all',mode = "deg"):
        if muni_list == "all":
            muni_list = self.muni_list
        coef = self.coef_df.loc[muni_list]
        if mode == "deg":
            return(coef)
        elif mode == "rad":
            return(np.deg2rad(coef))
        else:
            raise(ValueError("Invalid mode chosen. Chose from deg or rad"))