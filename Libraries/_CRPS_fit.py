# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 15:05:35 2018

@author: Tobias
"""
import pandas as pd
import numpy as np
import properscoring as ps
import scipy.optimize as opt
from datetime import time as d_time
from utilities import mk_list_dic, mk_coef_matrix_from_dic, d_time_to_quaters,\
return_to_root
import numba

# =============================================================================
# Evaluating score function
# ============================================================================
#@profile
def _mu_res(t_idx0,t_idx1,SPP_arr,WS_arr,WD_arr,alpha,beta,gamma,delta,\
                d_fc = d_time(1)):
    """ 
    Parameters
    ----------
    
    t_idx0 : int
        index in the ssp matrix (row wise)
    t_idx1 : int
        index in the forecast matrix (row wise)
    SPP_arr : numpy.ndarray
        SPP data for one or more muncipalities where the radiation model have 
        been substracted
    WS_arr : numpy.ndarray
        Wind speed data for one or more mincipalitees where model have been 
        substracted
    WD_arr : numpy.ndarray
        Wind direction data for one or more mincipalitees where model have been 
        substracted
    alpha,beta,gamma,delta : numpy.nddarray
        parameters for spp/ws, sin(WS), cos(ws)
    
    d_fc : datetime.time, optional
        The 
    """
#    if lag%4 != 0:
#        raise(NotImplementedError("Currently we can only predict with whole \
#                                  hour lags due to the fc resulution"))
    
    #get dimentions of different matriceis
    spp_k_max = alpha.shape[0] 
    ws_k_max = beta.shape[0]
    sinwd_k_max = gamma.shape[0]
    coswd_k_max = delta.shape[0]

    #handle indicies 
    lag = d_time_to_quaters(d_fc) #total number of quarters
    
    #These are the indicies of the maximum lag. This value depends on the
    #number of parameters included in the model is calculated with regards to
    #this
    spp_t_lag_max = t_idx0 - (spp_k_max - 1) - lag
    lag_fc = 0 #OBS: we know forecast well in advance so we dont have to add
    #lag to this set lag_fc = lag if we want lag back
    ws_t_lag_max = t_idx1 - 4*(ws_k_max - 1) - lag_fc # 4 for quarters in an hour
    sinwd_t_lag_max = t_idx1 - 4*(sinwd_k_max - 1) - lag_fc
    coswd_t_lag_max = t_idx1 - 4*(coswd_k_max - 1) - lag_fc
    
    
    #These are the indicies for the minimum lag. It corresponds to the lag
    #in the model. 1 is subtracted because of pythons way of indexing. 
    spp_t_lag_min = t_idx0 - (lag - 1)
    fc_t_lag_min =  t_idx1 - (lag_fc - 1) #will give the index +1
    spp_slice = SPP_arr[spp_t_lag_max:spp_t_lag_min]
    
    ws_slice = WS_arr[ws_t_lag_max:fc_t_lag_min][::4] #only one sample pr hour
    sinwd_slice = WD_arr[sinwd_t_lag_max:fc_t_lag_min][::4]
    coswd_slice = WD_arr[coswd_t_lag_max:fc_t_lag_min][::4]
    
    mu_r = fast_sum((spp_slice,ws_slice, sinwd_slice,coswd_slice),\
                    (alpha,beta,gamma,delta))
#    mu_r = 0
#    mu_r += np.sum(spp_slice*alpha)
#    mu_r += np.sum(ws_slice*beta)
#    mu_r += np.sum(np.sin(sinwd_slice)*gamma)
#    mu_r += np.sum(np.cos(coswd_slice)*delta)
    return(mu_r)

vec_full_mu_res = np.vectorize(_mu_res,\
                               excluded=['SPP_arr','WS_arr','WD_arr','alpha', \
                                         'beta','gamma','delta','d_fc'])

@numba.jit
def fast_sum(slices,coefs):
    alpha,beta,gamma,delta = coefs
    spp_slice,ws_slice, sinwd_slice,coswd_slice = slices
    mu_r = 0
    mu_r += np.sum(spp_slice*alpha)
    mu_r += np.sum(ws_slice*beta)
    mu_r += np.sum(np.sin(sinwd_slice)*gamma)
    mu_r += np.sum(np.cos(coswd_slice)*delta)
    return(mu_r)

def spatio_mod_res(rng,SPP_res,WS_res,WD_res,coef_arr,d_fc = d_time(1)):    
    """
    Calculates spatio temporal model for when time is for range rng currently 
    predicted an hour ahead. 
    OBS: remember to do simulation mode at some point
    
    Parameters
    ----------
    t : pandas.Timestamp
    """       
    #sigma parameter is not utilised therefore [:-1] 
    alpha,beta,gamma,delta = coef_arr[:-1]
    t_idx0 = np.in1d(SPP_res.index,rng).nonzero()[0]
    t_idx1 = np.in1d(WS_res.index,rng).nonzero()[0]
    mu_res = np.zeros(len(t_idx1))
    for i in range(len(t_idx1)):
        mu_res[i] = _mu_res(t_idx0[i],t_idx1[i],SPP_res.values,\
              WS_res.values,WD_res.values, alpha,beta,gamma,delta,\
              d_fc)
    #incomment later
    #mu_res = vec_full_mu_res(t_idx0,t_idx1,SPP_res.values,WS_res.values,\
#                         WD_res.values, alpha,beta,gamma,delta)
    return(mu_res)

def spatio_mod(rng,mu_r,rad_mod):    
    """
    Combines radiation model with the spatio temporal model and zeroes
    where necessary. write more documentation here
    """    
    #import min/max hours for model
    root = return_to_root()
    lib_path = 'Fortrolig_data/stem_data/'
    min_max_h = pd.read_pickle(root + lib_path + 'SPP_min_max_hours.pkl')
 
    spt_mod = mu_r + rad_mod
    spt_df = pd.DataFrame({'time':rng,'mod':spt_mod.values}).set_index("time")
    day_rng = pd.date_range(rng[0].date(),rng[-1].date(),freq = "D")
    for day in day_rng:
        daynext = day + pd.Timedelta(days = 1)
        
        #take out relevent hours in dataframe - the format causes some problems
        h0 = min_max_h.loc[day,'t0']
        h0 = (int(h0[:2]),int(h0[3:5])) #snip out hours and minutes
        h1 = min_max_h.loc[day,'t1']
        h1 = (int(h1[:2]),int(h1[3:5]))
        day_h0 = day + pd.Timedelta(hours = h0[0],minutes = h0[1])
        day_h1 = day + pd.Timedelta(hours = h1[0],minutes = h1[1])
        
        #find out where the indicies mathces on a given day
        idx_day = np.where(((spt_df.index >= day) & (spt_df.index <= day_h0)) \
                            | ((spt_df.index <= daynext) &\
                               (spt_df.index >= day_h1)))

        spt_df.iloc[idx_day] = 0   #the the value zero where this is the case
    
    #correct if the model becomes negative
    neg_index = np.where(spt_df < 0)
    spt_df.iloc[neg_index] = 0
    return(spt_df)

#@profile
def _vola_val(SPP_res,rng,d_fc = d_time(1)):
    """
    Calculates valatility value based of SPP residuals SPP_res. rng is a time
    index seris - a subset of the values of SPP_res. d_fc is the timelag used 
    in the forecast. 
    """
    k_max = 2 #cannot be changed currently
    SPP_arr_temp = SPP_res.values
    M = k_max*SPP_arr_temp.shape[1]
    sum_arr0 = (SPP_arr_temp[:-1] - SPP_arr_temp[1:])**2 #Get sum squared for time delay
    sum_arr1 = sum_arr0[:-1] + sum_arr0[1:] #Collects sums over rows with delay 1
    sum_arr2 = np.sum(sum_arr1,axis = 1) #sum over rows
    vola = np.sqrt(sum_arr2/M) #volatility value
    #[2:] becasue the two first times dont have enough avalable data
    vola_df = pd.DataFrame({'vola':vola,'idx':SPP_res.index[2:]}).\
    set_index('idx')
    vola_df_rng = vola_df.loc[rng - pd.Timedelta(hours = d_fc.hour,\
                                                 minutes = d_fc.minute)] 
    #incorpolate lag, and take out wanted values in rng
    N = vola_df_rng.shape[0]
    vola_val = vola_df_rng.values.reshape(N,)
    
    return(vola_val)



def _cbk(x0):
    """
    Callback funtion for optimizing algorithim - saves history and 
    prints progress. 
    """
    global count,hist
    hist[count] = S(x0)
    count += 1
#    print(coef_from_theta_df(x0))
    print(count,end=",")

class S_fam:
    """
    Class for handling mean score function using CRPS function as scorering 
    rule. Initialises with a lot of data and uses it it evaluate the mean
    score function. It also incorporates constraints for the parameter space
    Theta and the jacobian of these constrantints. 
    """
#    @profile
    def __init__(self,SPP_fit,SPP_res,WS_res,WD_res,x_len,coef,coef_len_list,\
                 d_fc = d_time(1),eps = 1e-2):
        """
        Parameters
        ----------
        x_len : int
          Length of the variabel
        coef : tuple with nested dic's
          Tuple with dictionaries of coefficients
        
        """
        SPP_arr = SPP_res.values
        WD_arr = WD_res.values
        WS_arr = WS_res.values
        self.SPP_fit = SPP_fit #spp in desired hours
        self.SPP_arr = SPP_arr
        self.WS_arr = WS_arr
        self.WD_arr = WD_arr
        self.SPP_index = SPP_res.index
        self.fc_index = WS_res.index
        self.M = SPP_arr.shape[1] #numbe of municipalities
        self.N = SPP_fit.shape[0]
        
        
        #setup timing
        self.rng =  np.in1d(self.SPP_index,self.SPP_fit.index).nonzero()[0]
        self.rng_fc = np.in1d(self.fc_index,self.SPP_fit.index).nonzero()[0]
        self.d_fc = d_fc
        
        #setup coefficients
        self.coef_arr = list(_coef_arr_from_coef(coef,flip = True)) 
        #support changing
        self.coef_len_list = coef_len_list
        #used for indexing transforming x0 into the right parts
        self.coef_len_cumsum = np.cumsum(np.hstack(self.coef_len_list))
        
        #Setup volatility values and constaints regarding that
        self.vola_val = _vola_val(SPP_res,self.SPP_fit.index,d_fc)
        self.vola_val_max = self.vola_val.max()
        self.jac_setup = self._jac_setup(x_len)
        self.cons = ({'type':'ineq', #forcing x[-2] + x[-1]*v >= eps
                      'fun' : lambda x: np.array([x[-2] + x[-1]\
                                                  *self.vola_val_max-eps]),
                      'jac' : lambda x: self.jac_setup[0]},

                     {'type':'ineq', #forcing x[-2] >= 0 
                      'fun' : lambda x: x[-2],
                      'jac' : lambda x: self.jac_setup[1]})
#    @profile
    def __call__(self,theta):
        self._coef_from_theta(theta) #put theta into ceof array
        mu = vec_full_mu_res(self.rng,self.rng_fc,SPP_arr = self.SPP_arr,\
                             WS_arr = self.WS_arr,WD_arr = self.WD_arr,\
                             alpha = self.coef_arr[0],\
                             beta  = self.coef_arr[1],\
                             gamma = self.coef_arr[2],\
                             delta = self.coef_arr[3],\
                             d_fc = self.d_fc)
        sig = self.coef_arr[4][0] + self.vola_val*self.coef_arr[4][1] #computes variance
        crps_vals = ps.crps_gaussian(self.SPP_fit,mu = mu, sig = sig)
        res = crps_vals.mean()
        return(res)
#    @profile
    def _coef_from_theta(self,x):
        #setup alpha
        chunks = np.split(x,self.coef_len_cumsum)
        for i in range(len(self.coef_len_list)): #loop parameter each parameter
            for m in range(self.M):
                coef_size = self.coef_len_list[i][m]
                if coef_size != 0:
                    self.coef_arr[i][:,m][-coef_size:] =\
                    np.flipud(chunks[self.M*i+m])
        self.coef_arr[4] = x[-2:]
            
    def _jac_setup(self,Len):
        """
        Jacobean matrix for constraint function
        """
        jac0 = np.zeros(Len)
        jac0[-2] = 1
        jac0[-1] = self.vola_val_max
        jac1 = np.zeros(Len)
        jac1[-2] = 1
        return((jac0,jac1))


# =============================================================================
# Optimization scripts
# =============================================================================
#@profile
def CPRS_fit(t_start,t_end,SPP_res, WD_res, WS_res, coef_len_list,
             muni_list = [849,851,860],muni = 851,d_fc = d_time(1),
             hour_fit =  ['04:00','20:00'],max_iter = 30, callback = False,
             return_vola_val = False):
    """
    Given data, muni info, forecast lag and some info about the model, it finds
    coefficients for the linear model:
        c(n,t) = u^r(n,t) such that c(n,t) ~ N(u^r(n,t),sig^2(n,t))
        where u^r(n,t) is a mean residual function after the radiation model and 
        sig^2(n,t) is the variance funtion. 
        
        u^r(n,t) = sum_{n}(sum_{lag}  alpha(n,lag)*u^r(n,t-lag)
                                    + beta(n,lag)*ws(n,t-lag)
                                    + gamma(n,lag)*sin(wd(n,t-lag))
                                    + delta(n,lag)*cos(wd(n,t-lag)))
        where u^r is residual SPP, ws is windspeed, wd is wind direction and
        n is municipality and t is time. 
        
        sig^2(n,t) = b0 + b1*v(t-1) 
        
        where v is volaitity value
    
    The continious rankes proberbility score (CRPS) is used and the mean 
    score function is used as an measure for goodnes of fit. The function
    is minimiced using the scipy minimize funtion with the
    Sequential Least Squares Problem (SLSQP) method.
        
    Parameters
    ----------
    t_start/t_end : pandas Timestamp
        start and end date for fitting. 
    SPP_res : pandas dataframe
        dataframe with residual SPP values
    WD_res : pandas dataframe
        dataframe with residual wind direction values
    WS_res : pandas dataframe
        dataframe with residual wind speed values
    muni_list : list/tuple 
        List of municipalites for wich the model is fitted for. 
    coef_len_list : list, tup
      List with nested lists with number of parameters for each mincipality and
      each type og parameter alpha/beta/gamma/delta
    muni : int
        Number of the muncipality whics is beeing optimized for. 
    d_fc : datetime.time, optional
        The time lag for which the model is beeing optimized for. Defaults to
        one hour. 
    hour_fit : list/tuple, optional
        Not all hours duing a day are desired optimice for. Set this argument
        in order to chose which hours of the day are decired to optimice for.
        Defaults to the hours 04:00 - 20:00
    max_iter : int, optional
        Maximum number of iterations the minimization algorithm should take.
        Defaults to 30
    callback : bool, optional
        The minimixation algotithm may print the count after each iteration
        and return out score history as result. Set to True if wanted. Defaults
        to False. 
    x0 : nun
  
    spp_k is the lag fittet for spp vales
    fc_k is the lag fitted for forecasts
    """
    #Setup range for evaluating score. 
    rng = pd.date_range(t_start,t_end,freq = "15min")
    rng_fit = rng[rng.indexer_between_time(hour_fit[0],hour_fit[1])]
    spp_res_fit = SPP_res[muni].loc[rng_fit]
    
    #Setup optimization
    coef = init_coef(muni_list,*coef_len_list,muni) #tuple with coef dic's
    x0 = _theta_from_coef(coef) #coefeficients to be altered
    global S #needs to be global in order for the callback function to work. 
    S = S_fam(spp_res_fit,SPP_res,WS_res,WD_res,len(x0),coef,coef_len_list,\
              d_fc = d_fc)
    if callback:
        global count, hist #is needed in order for callback function to work
        count = 0
        hist = np.zeros(max_iter+1)
        opt.minimize(S, x0,constraints= S.cons,\
                    method='SLSQP', options={'disp': True,'maxiter':max_iter},\
                                             callback = _cbk) 
        #writes to S instance
        if return_vola_val:
            return(hist[:count],S.coef_arr,S.vola_val)
        else:
            return(hist[:count],S.coef_arr)
    else:
        opt.minimize(S, x0,constraints= S.cons,\
                    method='SLSQP', options={'maxiter':max_iter})
        if return_vola_val:
            return(S.coef_arr,S.vola_val)
        else:
            return(S.coef_arr)

# =============================================================================
# Coefficient functions - transform vector to ordered coefficients
# =============================================================================
def init_coef(muni_list,n_alpha,n_beta,n_gamma,n_delta,muni):
    """
    Initialise coefficient dictionary for all parameters. 
    
    Parameters
    ----------
    muni_list : list,tuple
      List with muncipilaty numbers
    n_xxxxx : list,tuple
      List with length of of each coefficient arrays for all municipalities
      ex: n_alpha = [3,2,1] means muni0 have 3 parameters, muni1 have 2 
      parameters and so on. n_alpha is for SPP, n_beta is for wind speed,
      n_gamma is for sin(wind direction) and n_delta is for cos(wind direction)
    muni : int
      Number for the muni beeing modled. 
    """
    if not len(muni_list) == len(n_alpha) == len(n_beta) == len(n_gamma) == \
    len(n_delta):
        raise(ValueError("Not all tuples are same length - should be"))

    #initialise dictinonary for each parameter using mk_list_dic form utilities
    alpha = mk_list_dic(muni_list,n_alpha) 
    beta  = mk_list_dic(muni_list,n_beta)
    gamma = mk_list_dic(muni_list,n_gamma)
    delta = mk_list_dic(muni_list,n_delta)
    alpha[muni][0] = 1 #default alpha value - 
    b = [1,0] #default beta value
    return(alpha,beta,gamma,delta,b)
    


def _coef_arr_from_coef(coef,flip = False):
    """
    Given a list/tuple of dictionary coefficients, transform it into numpy
    matricies. It flip is True, then the rows of the matrix are flipped. 
    """
    alpha,beta,gamma,delta,b = coef
    alpha_arr = mk_coef_matrix_from_dic(alpha,flip)
    beta_arr  = mk_coef_matrix_from_dic(beta ,flip)
    gamma_arr = mk_coef_matrix_from_dic(gamma,flip)
    delta_arr = mk_coef_matrix_from_dic(delta,flip)
    return(alpha_arr,beta_arr,gamma_arr,delta_arr,b)

def _theta_from_coef(coef):
    """
    Given a coefficient list/tuple, transform into one vetor with coefficients.
    Jonas jeg giver kage hvis du finder denne grimme funktion! 
    """
    theta = []
    #last index is variance parameter and is appended differently
    for element in coef[:-1]:
        for key in element:
            for list_element in element[key]:
                theta.append(list_element)
    theta = np.hstack((theta,coef[-1])) #last index
    return(theta)

def _theta_size_from_coef_arr(coef_arr):
    """
    Given a coefficient list transform into one vetor with coefficients.
    """
    theta_len = 0
    #last index is variance parameter and is appended differently
    for element in coef_arr:
        theta_len += element.size - np.sum(element == 0) #subtract 0 elements
    return(theta_len - 2) #2 is suntracted in order for the sigma parameters
    #not to matter
    
def _theta_from_coef_arr(coef_arr,coef_len_list,new_para_idx):
    """
    Given a coefficient list transform into one vetor with coefficients.
    Not currenyly used
    """
    coef_arr[new_para_idx[0]][new_para_idx[1]][new_para_idx[2]] = 1
    a_sh, b_sh,g_sh,d_sh  = coef_arr[0].size,coef_arr[1].size,\
                            coef_arr[2].size,coef_arr[3].size
    alpha_list = np.flipud(coef_arr[0]).T.reshape(a_sh,)
    beta_list = np.flipud(coef_arr[1]).T.reshape(b_sh,)
    gamma_list = np.flipud(coef_arr[2]).T.reshape(g_sh,)
    delta_list = np.flipud(coef_arr[3]).T.reshape(d_sh,)
    theta = np.hstack((alpha_list,beta_list,gamma_list,delta_list,coef_arr[-1]))
    theta = theta[np.nonzero(theta)]
    return(theta)
