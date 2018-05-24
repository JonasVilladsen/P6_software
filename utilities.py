# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 10:26:34 2018

@author: Martin Kamp Dalgaard & Tobias Kallehauge
"""

import os
import pandas as pd
import numpy as np
import time

root = os.getcwd()[:-24]

def return_to_root(n_max = 10): # Return to the root directory in svn from a subfolder
    root = os.getcwd()
    for i in range(n_max):
        if root[-3:] != "svn":
            os.chdir("..")
            root = os.getcwd()
        else:
            break
    if root[-3:] != "svn":
        raise(OSError("\"svn\" not found using recursion.\nCheck if script is in a subfolder under \"svn\".\nIf this is the case set n_max higher."))
    root += "/"
    return root

def choose_days_from_timerange(timerange,days):
    """
    Sort out dates not in datelist and return sorted timerange.
    
    Input:
        timerange: pandas date_range
        days: pandas date_range with frequency at least "D" or greater.
        
    Return:
        Return pandas date_range
        
    ------
    Example
    t0 = pd.Timestamp(2017,1,1); t1 = pd.Timestamp(2017,2,1)
    range1 = pd.date_range(t0,t1,freq = '15min')
    range2 = pd.date_range(t0,t1,freq = '2D')
    range_sorted = choose_days_from_timerange(range1,range2)
    """
    M = len(timerange)
    new_rng = []
    j = 0
    for i in range(M):
        if timerange[i].date() in days:
            new_rng.append(timerange[i])
            j += 1
    return pd.DatetimeIndex(new_rng)


def load_convert_muni_to_grid():
    root = return_to_root()
    stem_path = "Fortrolig_data/stem_data/"
    conv_path = stem_path + "Kommune_GridNr.xlsx"
    conv_sheet = pd.read_excel(root + conv_path)
    conv_sheet.set_index('Kommune_Nr',inplace= True) #Let kommune_nr be index
    return(conv_sheet)

def muni_list_to_grid_list(muni_list = 'all'):
    """"
    Converts list of muni's to grid list. Input must be of array type
    """
    conv_sheet = load_convert_muni_to_grid()
    if not set(muni_list).issubset(set(conv_sheet.index)) and muni_list != 'all':
        raise(ValueError("Element in muni_list is invalid municipilaty number."))

    if muni_list == "all":
        muni_list = conv_sheet.index
    conv_sheet = conv_sheet.loc[muni_list] #get grid points for chosen munis    
    
    N = np.shape(conv_sheet)[0] #elements in dataframe
    grid_list = np.array(conv_sheet[['GNr1','GNr2','GNr3']]).reshape(3*N)
    grid_list = mk_unique_no_nan_int(grid_list)
    return(list(grid_list),conv_sheet)

def zeropad_hourstring(string):
    if len(string) == 1:
        return "0" + string
    else:
        return string

def mk_unique_no_nan_int(array,dtype = "int16"):
    nan_idx = np.isnan(array) #where Truth/False if value is nan
    idx = np.argwhere(nan_idx == True)
    array_new = np.delete(array,idx) #array with no nan
    array_new = np.array(np.unique(array_new),dtype = "int16") #remove repeats
    return(array_new)
    
def mk_list_dic(keys,len_list):
    """
    Make dictionary for given keys with numpy zero arrays where len_list is a
    tuple with the length of the array for each key
    """
    N = len(len_list)
    return({keys[i] : np.zeros(len_list[i]) for i in range(N)})

def max_list_len_dic(dic):
    """
    Finds length of dictionary with containers and return the length of the 
    largest one
    """
    return(max([len(i) for i in dic.values()]))
    
def mk_coef_matrix_from_dic(dic,flip = True):
    """
    Creates numpy array with coefficients made from dictionary. Used in 
    spatio temporal model. Flip argument flips the arrays
    """
    keys = list(dic.keys())
    M = len(keys) #number of municaplites
    dic_N = max_list_len_dic(dic)
    
    dic_arr = np.zeros((dic_N,M)) 
    for nr in range(M): #loop over each muncipality
        for lag in range(len(dic[keys[nr]])):
            dic_arr[:,nr][lag] = dic[keys[nr]][lag]
    if flip:
        dic_arr = np.flipud(dic_arr)
    return(dic_arr)

def d_time_to_quaters(t):
    """
    Converts a datetime object into total number of quarters. Can only handle
    whone number of minutes. 
    """
    return(4*t.hour + int(t.minute/15))
    
