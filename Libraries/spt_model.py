# -*- coding: utf-8 -*-
"""
Created on Sun May  6 18:46:51 2018

@author: Tobias
"""

import _spt_setup as setup
import _CRPS_fit as CRPS
from model_evaluation import BIC,BIC_edit
import pandas as pd
import numpy as np
import copy
from datetime import time as d_time
from utilities import return_to_root

class spt_mod:
    """
    Class for handling spatio temporal model. Write more here
    """
    
    def __init__(self,t_start = pd.Timestamp(2017,1,1),\
                 t_end = pd.Timestamp(2017,12,31),muni_list = [849,851,860]):
        
        hours = ["00:00","23:45"] #import for all hours
        self.muni_list = np.asarray(muni_list)
        self.muni_list.sort() #sort muni list if it is not
        self.SPP, self.rad_vals, self.SPP_res, self.WS_res, self.WD_res = \
        setup._setup(t_start,t_end,muni_list,hours)
        self.rng = self.SPP.index
        self.hour_fit_df = self.load_hour_fit()
        self.M = self.muni_list.size
    
    def spt_fit_scalar(self,t,muni,d_fc = d_time(1),window_length = 10,\
                       max_iter = 50,coef_lim = 10,\
                       print_progress = False,
                       BIC_mode = 'var_sig'):
        """
        Calculates coefficients for spatio temporal model for one time and for 
        one delta_fc using scoring rules and fitting with the BIC for model 
        selection. 
        
        Parameters
        ----------
        
        t : pd.Timestamp
            Time what time it is. Ex pd.Timestamp(2017,5,4,12) can acces the 
            data up till 12:00 04/05-2012 
        muni : int
            Municipality number for the munipality beeing optimiced for. 
        d_fc : datetime.time
            Delta forecast, how far into the future should be predicted. 
            Ex d_fc = d_time(1) will predict 1 hour into the future
        windiow_length, int
            How long in the past the model should be fitted with. Ex 
            window_length = 45 will use a window from that point and 45 days
            into the past.
        max_iter : int, optional
            Maxmimum number of iterations the fitting function will utilize. 
        coef_lim : int, optional
            Maxmimum number of time lag coefficients any coefficient can have.
        BIC_mode : str, optional
            BIC can be calculated under different assumptions about the model
            for 'classic' mode it uses the maximum likelihood assuming constant
            varaince and for 'var_sig' it uses the volatitity value to compute
            the maximum likelihood value. 
            
        Returns
        -------
        coef : Coefficients for the model with information about what the model
        order became. 
        """
        if t - pd.Timedelta(days = window_length) < self.rng[0]:
            raise(ValueError("Make sure that t - day_range > %s"\
                             %str(self.rng[0])))
        if d_fc.hour + d_fc.minute/60 > 6:
            raise(ValueError("Make sure that d_fc < 6H"))
        
        if not BIC_mode in ('classic','var_sig'):
            raise(ValueError("BIC_mode should be either 'classic' or 'var_sig'"))
        
        #implement some sanity checks
        
        #set some preferences
        self.coef_lim = coef_lim
        self.max_iter = max_iter
        self.print_progress = print_progress
        self.BIC_mode = BIC_mode
        #setup timing
        t_past_max = t - pd.Timedelta(days = window_length)
        
        #start with one parameter for the chosen muncipality and build up
        muni_idx = np.where(self.muni_list == muni)[0][0]
        n_alpha = [0 for i in range(self.M)]
        n_alpha[muni_idx] = 1;
        n_beta =  [0 for i in range(self.M)]
        n_gamma = [0 for i in range(self.M)]
        n_delta = [0 for i in range(self.M)]
        self.coef_nr_list = (n_alpha,n_beta,n_gamma,n_delta)
        hours_fit = self._hour_fit(t_past_max,t)
        
        coef_0,vola_val = CRPS.CPRS_fit(t_past_max,t,
                               self.SPP_res.loc[t_past_max.date():t],
                               self.WD_res.loc[t_past_max.date():t],
                               self.WS_res.loc[t_past_max.date():t],
                               self.coef_nr_list,
                               muni_list = self.muni_list,muni = muni, 
                               hour_fit = hours_fit, max_iter = max_iter,
                               callback = False,d_fc = d_fc,
                               return_vola_val = True)
        
        self.current_vola_val = vola_val
        #used for computing BIC-mode
        
        #fit for SPP coefficients
        if self.print_progress:
            print("Fitting SPP")
        coef_arr = self._fit_for_information(0,coef_0,d_fc,muni_idx,\
                                         t_past_max,t,hours_fit)
    
        if self.print_progress:
            print("\nFitting WS")
        coef_arr = self._fit_for_information(1,coef_arr,d_fc,muni_idx,\
                                         t_past_max,t,hours_fit)
        
        #fit for cos(WD) coefficients
        if self.print_progress:
            print("\nFitting cos(WD)")
        
        coef_arr = self._fit_for_information(2,coef_arr,d_fc,muni_idx,\
                                                 t_past_max,t,hours_fit)
        #fit for sin(WD) coefficients
        coef_arr = self._fit_for_information(3,coef_arr,d_fc,muni_idx,\
                                                 t_past_max,t,hours_fit)
        return(coef_arr)

        
    def _fit_for_information(self,coef_idx,coef_arr,d_fc,muni_idx,\
                             t_start,t_end,hours_fit):
        """
        Give type of indepenet variable fit model for thouse parameters
        using bic. Ex for coef_idx = 1 it fits for number of SPP parameters
        for different muncipalities
        """
        muni = self.muni_list[muni_idx]
        #put fitting muni in front of list with numpy rool
        muni_sorted_list = np.roll(self.muni_list,-muni_idx) 
        BIC0 = self._BIC_model(t_start,t_end,coef_arr,hours_fit,d_fc,muni,
                               self.current_vola_val)
        for muni_loop in muni_sorted_list: #loop over municipalities
            loop_muni_idx = np.where(self.muni_list == muni_loop)[0][0]
            coef_nr_list_copy = copy.deepcopy(self.coef_nr_list)
            for n in range(self.coef_lim): #loop for each index
                if n > 4 and coef_idx > 0: #there is some problems with
                    #acceing weather forecast more than 4 hours into the past
                    break
                if self.print_progress:
                    print(muni_loop, n, end = ",")
                coef_nr_list_copy[coef_idx][loop_muni_idx] += 1 #add one parameter
                #fit the model
                coef_arr_temp,vola_val = CRPS.CPRS_fit(t_start,t_end,
                               self.SPP_res.loc[t_start.date():t_end],
                               self.WD_res.loc[t_start.date():t_end],
                               self.WS_res.loc[t_start.date():t_end],
                               coef_nr_list_copy,
                               muni_list = self.muni_list,muni = muni, 
                               hour_fit = hours_fit, max_iter = self.max_iter,
                               callback = False,d_fc = d_fc,
                               return_vola_val = True)
                self.current_vola_val = vola_val

                BIC_mod = self._BIC_model(t_start,t_end,coef_arr_temp,\
                                      hours_fit,d_fc,muni,
                                      self.current_vola_val)
#                print("Temp coef:"); print(coef_arr_temp[coef_idx],end = "\n")
                if self.print_progress:
                    print("BIC old = %.2f, BIC new %.2f" %(BIC0,BIC_mod))
                if BIC_mod <= BIC0: #add paramter if BIC is lowered
                    self.coef_nr_list = copy.deepcopy(coef_nr_list_copy)
                    BIC0 = BIC_mod
                    coef_arr = coef_arr_temp
#                    print("\nYes you added a parameter!!!\n")
                else: #else stop adding parameter
                    break
        return(coef_arr)
        #start with ht muncipality beeing optimized for
        
    def spt_fit_sclaer_w_coefnr(self,t,muni,coef_len_list,d_fc = d_time(1),\
                                window_length = 10,max_iter = 50,\
                                coef_lim = 10,print_progress = False):
        """
        Fit spatio temporal model at time t by setting amount of parameters 
        manually with coef_nr_list
        
        Parameters
        ----------
        coef_len_lisr: list, tup
          List with nested lists with number of parameters for each mincipality and
          each type og parameter alpha/beta/gamma/delta
         see ?spt_fit_scalar for more info about parameters
        """
        if t - pd.Timedelta(days = window_length) < self.rng[0]:
            raise(ValueError("Make sure that t - day_range > %s"\
                             %str(self.rng[0])))
        if d_fc.hour + d_fc.minute/60 > 6:
            raise(ValueError("Make sure that d_fc > 6"))
        
        self.coef_lim = coef_lim
        self.max_iter = max_iter
        self.print_progress = print_progress
        #setup timing
        t_past_max = t - pd.Timedelta(days = window_length)
        
        #start with one parameter for the chosen muncipality and build up
        hours_fit = self._hour_fit(t_past_max,t)
        
        coef_arr = CRPS.CPRS_fit(t_past_max,t,\
                               self.SPP_res.loc[t_past_max.date():t],\
                               self.WD_res.loc[t_past_max.date():t],\
                               self.WS_res.loc[t_past_max.date():t],\
                               coef_len_list,\
                               muni_list = self.muni_list,muni = muni, \
                               hour_fit = hours_fit, max_iter = max_iter,\
                               callback = False,d_fc = d_fc)
        return(coef_arr)


    def spt_fit_update_coef(self,t_start,t_end,muni,d_fc = d_time(1),\
                            window_length = 10,\
                            max_iter = 50,coef_lim = 5,print_progress = False,\
                            update_every = 'H',start_hour = None):
        """
        Get coefficients spatio temporal model in the interval t_start,t_end
        and updating the coefficients every d_time
        
        Parameters
        ----------
        t_start,t_end : pd.Timestamp
            Start/end time for fitting coefficients
        update_every: pandas freq format
            How often parameters should be updated. ex. 'H' for every hours
        See ?spp_fit_scalar for more info
        """
        fit_rng = pd.date_range(t_start,t_end,freq = update_every)
        if not isinstance(start_hour,type(None)):
            fit_rng += pd.Timedelta(hours = start_hour.hour,\
                                    minutes = start_hour.minute) #add 12 hours if needed
        #remove hours where sun has set from range
        hour_fit = self._hour_fit(t_start.date(),t_end.date())
        fit_rng = fit_rng[fit_rng.indexer_between_time(hour_fit[0],hour_fit[1])]
        coef_dic = {}
        if print_progress:
            print("Moddel will be fitted for the following intervals:")
            print(fit_rng)
        for t in fit_rng:
            if print_progress:
                print("%s, " %str(t.time()),end="")
            coef_dic[t] =  self.spt_fit_scalar(t,muni,d_fc = d_fc,\
                        window_length = window_length,max_iter = max_iter,\
                        coef_lim = coef_lim,print_progress = False)
        return(coef_dic)
        
    def spt_fit_update_d_fc(self,t_start,t_end,muni,\
                            window_length = 10,\
                            max_iter = 50,coef_lim = 5,print_progress = False,\
                            d_fc_freq = '15min'):
        """
        Get coefficients spatio temporal model in the from t_start up to
        t_end with a resulution of d_fc_freq
        
        Parameters
        ----------
        t_start: pd.Timestamp
            Start/end time for fitting coefficients
        d_fc_freq: pandas freq format in min
            How often parameters should be updated. ex. 'H' for every hours
        See ?spp_fit_scalar for more info
        """
        if d_fc_freq[-3:] != 'min':
            raise(NotImplementedError("d_fc_freq should be in minute format"))
        rng_fit = pd.date_range(t_start,t_end,freq = d_fc_freq)
        d_fc_int = eval(d_fc_freq[:2])
        #different fc with datetime format
        d_fc_rng = [d_time(hour = i*d_fc_int//60,minute = i*d_fc_int%60) 
                    for i in range(1,len(rng_fit))]
        coef_dic = {}
        if print_progress:
            print("Moddel will be fitted for the following intervals:")
            print(rng_fit[1:])
        for i in range(len(d_fc_rng)):
            if print_progress:
                print("%s, " %str(rng_fit[i+1].time()),end="")
#            try:
            coef_dic[rng_fit[i+1]] =  self.spt_fit_scalar(t_start,muni,
                    d_fc = d_fc_rng[i],window_length = window_length,
                    max_iter = max_iter, coef_lim = coef_lim)
#            except:
#                coef_dic[rng_fit[i+1]] = np.nan
#                print("Failed %s" %str(rng_fit[i+1]),end="")
        return(rng_fit[1:],d_fc_rng,coef_dic)
#        
    def load_hour_fit(self):
        root = return_to_root()
        lib_path = 'Fortrolig_data/stem_data/'
        min_max_h = pd.read_pickle(root + lib_path + 'SPP_min_max_hours.pkl')
        return(min_max_h)
        
    def _hour_fit(self,date0,date1):
        """
        Given two dates find the minimum and maximum hours where the sun 
        rises/sets.
        """
        h0 = np.min(self.hour_fit_df.loc[date0:date1])['t0']
        h1 = np.max(self.hour_fit_df.loc[date0:date1])['t1']
        h0 = d_time(int(h0[:2]),int(h0[3:5])) #snip out hours and minutes
        h1 = d_time(int(h1[:2]),int(h1[3:5]))
        return(h0,h1)
        
    def _BIC_model(self,t_start,t_end,coef_arr,hours,d_fc,muni,vola_val):
        """
        Calculates the BIC for a model with the given coefficients in the range
        [t_start,t_end]
        """
        rng,mu_t = self.spt_model_const_coef(t_start,t_end,coef_arr,
                                             d_fc,muni,hours)
        k = CRPS._theta_size_from_coef_arr(coef_arr) #model order
        if self.BIC_mode == 'classic': 
            #compute sigma for equally distributed samples
            BIC_mod = BIC(self.SPP[muni].loc[t_start:t_end].loc[rng].values
                              ,mu_t.values,k)
        elif self.BIC_mode == 'var_sig': #take varing sigma into account
            BIC_mod = BIC_edit(self.SPP[muni].loc[t_start:t_end].loc[rng].values
                          ,mu_t.values,k,vola_val,coef_arr[-1])
            #coef_arr[-1] is the b coefficient. 
        return(BIC_mod)
    
    def spt_model_const_coef(self,t_start,t_end,coef_arr,d_fc,muni,\
                             hours = 'all',hour_buff = False):
        """
        Gets the spatio temporal model in the range [t_start,t_end] with some
        forecast delta for for a muni.
        
        Parameters
        ----------
        t_start/t_end : pd.Timestamp
            Begin and end time for model
        coef_arr : list/tuple with nested numpy.ndarrays
            Coefficients for the model - are kept constant fo the period
        hours : (datetime.time,datetime.time)
            If less hours whan the whole days is wanted set this as datetime
            format
        d_fc : datetime.time 
            How many hours into the futue should be forecasted
        muni : int
            Municipality beeing fitted for
        
        Returns
        -------
        rng : pd.date_range
            range the model have been fitted for
        mu_t : pd.DataFrame
            Dataframe with fitted values
        """
        rng = pd.date_range(t_start,t_end,freq = "15min")
        if hours != 'all':
            if hour_buff:
                buffer = 15
                h0_h = hours[0].hour; h0_min = hours[0].minute
                h1_h = hours[1].hour; h1_min = hours[1].minute
                h0 = pd.Timedelta(hours = h0_h, minutes  = h0_min - buffer) #add a small buffer
                h1 = pd.Timedelta(hours = h1_h, minutes  = h1_min + buffer) #add a small buffer
                h0 = d_time(hour = h0.components.hours,
                            minute =  h0.components.minutes)
                h1 = d_time(hour = h1.components.hours,
                            minute =  h1.components.minutes)
            else:
                h0, h1 = hours[0],hours[1]
            rng = rng[rng.indexer_between_time(h0,h1)]
        else:
            hours = ["00:00","23:00"] #there is an import bug i don't want to fix
            rng = rng[rng.indexer_between_time(hours[0],hours[1])]
        mu_r = CRPS.spatio_mod_res(rng,self.SPP_res,self.WS_res,self.WD_res,\
                                   coef_arr,d_fc)
        mu_t = CRPS.spatio_mod(rng,mu_r,self.rad_vals[muni].loc[rng]) #total mean
        return(rng,mu_t)
    
    def spt_model_updating_d_fc(self,rng_fit,d_fc_rng,coef_dic,muni,
                                hours='all'):
        """
        Gets the spatio temporal model in the range [t_start,t_end] with some
        forecast delta for for a muni. The coefficients can be updated as 
        dictated by the coef_dic - coefficients dictionary with info on when
        coefficients should be counts.
        
        Parameters:
        ----------
        coef_dic : dic
            dictionary with info about when to update coefficients. 
        For info about rest of the parameters see ?spt_model_const_coef 
        """
        #setup up intervalst from coefficient dictionary
        mu_t = pd.DataFrame()
        for i in range(len(rng_fit)):
            rng,mu_temp = self.spt_model_const_coef(rng_fit[i],rng_fit[i],\
                                                    coef_dic[rng_fit[i]],\
                                                d_fc = d_fc_rng[i], 
                                                muni = muni,\
                                                hours = hours)
            mu_t = mu_t.append(mu_temp) #add to df and leave out last
            #index
        return(mu_t.index,mu_t)
        
    def spt_model_updating_coef(self,t_start,t_end,coef_dic,d_fc,muni,
                                hours='all'):
        """
        Gets the spatio temporal model in the range [t_start,t_end] with some
        forecast delta for for a muni. The coefficients can be updated as 
        dictated by the coef_dic - coefficients dictionary with info on when
        coefficients should be counts.
        
        Parameters:
        ----------
        coef_dic : dic
            dictionary with info about when to update coefficients. 
        For info about rest of the parameters see ?spt_model_const_coef 
        """
        if not isinstance(muni,(int,np.int)):
            raise(TypeError("muni argument should be a n integer"))
        if not isinstance(d_fc,d_time):
            raise(TypeError("d_fc should be datetime.time type"))
        #setup up intervalst from coefficient dictionary
        t_coef = list(coef_dic.keys())
        rng_fit = np.hstack((t_start,t_coef[1:],t_end))
        mu_t = pd.DataFrame()
        for i in range(len(rng_fit)-1):
            key = t_coef[i]
            rng,mu_temp = self.spt_model_const_coef(rng_fit[i],rng_fit[i+1],\
                                                    coef_dic[key],\
                                                d_fc = d_fc, muni = muni,\
                                                hours = hours)
            mu_t = mu_t.append(mu_temp.iloc[:-1]) #add to df and leave out last
            #index
        return(mu_t.index,mu_t)