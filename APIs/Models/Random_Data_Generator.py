# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 12:46:55 2021

Authors
Cailong Hua (hua00023@umn.edu)
Sivaraman Rajaganapathy (sivrmn@umn.edu)

Warning: For research use only
"""


# =============================================================================
# Description
# =============================================================================

# Code for simulating dynamic models using Hamiltonians

# =============================================================================
 
 
# =============================================================================
# Imports
# ============================================================================= 
import os
import numpy as np
import scipy as sc
import scipy.stats as scstats
from scipy.stats import norm as scnorm
from scipy.optimize import minimize
from scipy.stats import beta as scbeta

import sys
sys.path.append('../../')

# =============================================================================
#%%


# =============================================================================
# Class containing functions for generating work distributions
# =============================================================================
class Random_Data_Generator():
    
    #--------------------------------------------------------------------------        
    # Init
    # **T:** Temperature in Kelvin
    #--------------------------------------------------------------------------  
    def __init__(self, T): 
        
        self.T = T
        
        self.kb = sc.constants.Boltzmann
        
        self.beta = 1/(self.kb*self.T)
        
        self.create_shaping_params()
        
        self.optim_var = 0
    #--------------------------------------------------------------------------  
    
    
    
    #--------------------------------------------------------------------------        
    # Get Truncated Normal samples
    #--------------------------------------------------------------------------  
    def get_truncated_normal(self, low_lim, high_lim, mean = 0, std = 1, size = 100): 
    
        rv_trunc = scstats.truncnorm((low_lim - mean) / std, (high_lim - mean) / std, loc=mean, scale=std)
        
        x = rv_trunc.rvs(size)

        return(x)
    #--------------------------------------------------------------------------
    
    
    #--------------------------------------------------------------------------
    # Bimodal distribution by dual truancated symmetric Gaussians
    #--------------------------------------------------------------------------
    def get_bimodal_sym_trunc_normal(self, low_lim, mean_a, mean_b, std_a, std_b, w_a = 0.5, w_b = 0.5, d = 0, size = 100):
        
        size_a = int((w_a/(w_a+w_b))*size)
        size_b = int(size - size_a)
        
        low_lim_a = low_lim
        high_lim_a = 2*mean_a - low_lim_a
        
        # low_lim_b = high_lim_a + d
        # high_lim_b = 2*mean_b - low_lim_b
        
        delta = mean_a - low_lim_a
        low_lim_b = mean_b - delta
        high_lim_b = mean_b + delta
        
        rv_a = self.get_truncated_normal(low_lim_a, high_lim_a, mean = mean_a, std = std_a, size = size_a)
        
        rv_b = self.get_truncated_normal(low_lim_b, high_lim_b, mean = mean_b, std = std_b, size = size_b)
        
        rv = np.concatenate((rv_a, rv_b))    
        
        return(rv)
    #--------------------------------------------------------------------------
    
    
    #--------------------------------------------------------------------------        
    # Get Moments for truncated Normal distribution
    # See: "Wang, Yibing, Wei Dong, Liangqi Zhang, David Chin, Markos Papageorgiou, Geoffrey Rose, and William Young. "Speed modeling and travel time estimation based on truncated normal and lognormal distributions." Transportation research record 2315, no. 1 (2012): 66-72."
    # **Arguments:**
    # **low_lim:** Lower limit for the underlying truncated normal
    # **high_lim:** Hiher limit for the underlying truncated normal
    # **true_x_mean:** Mean of the underlying untruncated normal distribution
    # **true_x_std:** Standard deviation of the underlying untruncated normal distribution    
    #--------------------------------------------------------------------------  
    def get_moments_trunc_normal(self, low_lim, high_lim, true_x_mean, true_x_std): 

        """
        Get Moments for truncated Normal distribution
        See: "Wang, Yibing, Wei Dong, Liangqi Zhang, David Chin, Markos Papageorgiou, Geoffrey Rose, and William Young. "Speed modeling and travel time estimation based on truncated normal and lognormal distributions." Transportation research record 2315, no. 1 (2012): 66-72."
        **Arguments:**
        **low_lim:** Lower limit for the underlying truncated normal
        **high_lim:** Hiher limit for the underlying truncated normal
        **true_x_mean:** Mean of the underlying untruncated normal distribution
        **true_x_std:** Standard deviation of the underlying untruncated normal distribution    
                     
        """
        a = low_lim
        b = high_lim        

        mu = true_x_mean
        sig= true_x_std
        
        
        a0 = (a - mu)/sig
        b0 = (b - mu)/sig
        
        T = scnorm.cdf(b0) - scnorm.cdf(a0)
    
        # Equation 15         
        num = sig*(scnorm.pdf(a0) - scnorm.pdf(b0))         
        trunc_x_mean = (num/T)  + mu
        

        # Equation 16
        num1 = (sig**2)*(a0*scnorm.pdf(a0) - b0*scnorm.pdf(b0))

        num2 = (sig**2)*(scnorm.pdf(a0) - scnorm.pdf(b0))**2

        trunc_x_var = (sig**2) + num1/T - num2/(T**2)

        trunc_x_std = np.sqrt(trunc_x_var)

        return(trunc_x_mean, trunc_x_std)
    #--------------------------------------------------------------------------
        
    
    #--------------------------------------------------------------------------        
    # Get Moments for truncated Log Normal distribution
    # See: "Wang, Yibing, Wei Dong, Liangqi Zhang, David Chin, Markos Papageorgiou, Geoffrey Rose, and William Young. "Speed modeling and travel time estimation based on truncated normal and lognormal distributions." Transportation research record 2315, no. 1 (2012): 66-72."
    # **Arguments:**
    # **low_lim:** Lower limit for the underlying truncated normal
    # **high_lim:** Hiher limit for the underlying truncated normal
    # **true_x_mean:** Mean of the underlying untruncated normal distribution
    # **true_x_std:** Standard deviation of the underlying untruncated normal distribution
    #--------------------------------------------------------------------------  
    def get_moments_trunc_lognormal(self, low_lim, high_lim, true_x_mean, true_x_std): 
    
        a = np.exp(low_lim)
        b = np.exp(high_lim)        

        mu = true_x_mean
        sig= true_x_std

        true_y_mean = np.exp(mu + (sig**2)/2)
        true_y_var = (np.exp(sig**2)-1)*(np.exp(2*mu + sig**2))
        true_y_std = np.sqrt(true_y_var)
    

        a0 = ( np.log(a) - mu ) / sig
        b0 = ( np.log(b) - mu ) / sig
        
        num = scnorm.cdf(-sig + b0) - scnorm.cdf(-sig + a0)
        den = scnorm.cdf(b0) - scnorm.cdf(a0)
        trunc_y_mean = true_y_mean *  num/den
    
        num1 = scnorm.cdf(-2*sig + b0) - scnorm.cdf(-2*sig + a0)
        var1 = num1 / den - (num**2)/(den**2)
        var2 = num1 / den
        trunc_y_var = (true_y_mean**2) * var1 + (true_y_std**2)*var2
        trunc_y_std = np.sqrt(trunc_y_var)
    
    
        return (trunc_y_mean, trunc_y_std)
    #--------------------------------------------------------------------------
    
    
    #--------------------------------------------------------------------------        
    # Get Moments for log Uniform distribution
    # **Arguments:**
    # **low_lim:** Lower limit for the underlying uniform distribution
    # **high_lim:** Hiher limit for the underlying uniform distribution
    #--------------------------------------------------------------------------  
    def get_moments_log_uniform(self, low_lim, high_lim): 
        
        
        a = np.exp(low_lim)
        b = np.exp(high_lim)
        
        y_mean = (b-a)/np.log(b/a)
        
        val1 = (b**2 - a**2)/(2*np.log(b/a))
        val2 = ((b-a)/(np.log(b/a)))**2
        
        y_var = val1 - val2
        
        y_std = np.sqrt(y_var)
        
        return (y_mean, y_std)
    #--------------------------------------------------------------------------        
    
    
    #--------------------------------------------------------------------------	
    #	Create shaping parameters for different distributions
    #--------------------------------------------------------------------------        
    def create_shaping_params(self):
        
        # For all distributions
        self.N_wd = 1000000 #No. of samples used for the wd distribution optimization process
        self.wd_mean = 8

        # Beta distribution shape params
        self.beta_dist_a = 0.1
        self.beta_dist_b = 0.5
        
        # Bimodal distribution (via dual symmetric truncated normals) shape params
        self.bimod_low_lim = -3
        
        self.bimod_mean_a = 0
        self.bimod_mean_b = 10
        
        self.bimod_w_a = 0.5
        self.bimod_w_b = 0.5
        
        self.bimod_d = 0
        
        return()	
    #--------------------------------------------------------------------------    
    
    
    
    #--------------------------------------------------------------------------	
    #	Sample based optimization function for the truncated normal distribution
    #--------------------------------------------------------------------------        
    def optim_samp_trunc_norm_wd(self, wd_low):
        
        wd_high = 2*self.wd_mean - wd_low
        wd_samp = self.get_truncated_normal(wd_low, wd_high, self.wd_mean, self.wd_std, size = self.N_wd)
        mean_exp_wd = np.mean(np.exp(-wd_samp))
        err = np.abs(mean_exp_wd - 1)
        
        return(err)	
    #--------------------------------------------------------------------------    


    #--------------------------------------------------------------------------	
    #	Sample based optimization function for the beta distribution
    #--------------------------------------------------------------------------        
    def optim_samp_beta_wd(self, wd_low):
         
        a = self.beta_dist_a
        b = self.beta_dist_b
        wd_samp = scbeta.rvs(a, b, size = self.N_wd) 
        wd_high = ((a+b)/a)*(self.wd_mean - wd_low) + wd_low
        wd_samp = wd_samp*(wd_high - wd_low) + wd_low
        mean_exp_wd = np.mean(np.exp(-wd_samp))
        err = np.abs(mean_exp_wd - 1)
        
        return(err)	
    #--------------------------------------------------------------------------    


    #--------------------------------------------------------------------------	
    #	Sample based optimization function for the bimodal distribution composed of
    # two truncated symmetric normals
    #--------------------------------------------------------------------------        
    def optim_samp_bimodal_sym_wd(self, wd_std):
         
        std_a = wd_std
        std_b = wd_std
        
        wd_samp = self.get_bimodal_sym_trunc_normal(self.bimod_low_lim, self.bimod_mean_a, self.bimod_mean_b, 
                                               std_a, std_b, w_a = self.bimod_w_a, w_b = self.bimod_w_b, 
                                               d = self.bimod_d, size = self.N_wd)
            
        mean_exp_wd = np.mean(np.exp(-wd_samp))
        err = np.abs(mean_exp_wd - 1)
        
        return(err)	
    #--------------------------------------------------------------------------    


    #--------------------------------------------------------------------------	
    #	Function to set shape parameters
    #--------------------------------------------------------------------------        
    def set_shape_params(self, wd_mean, N_wd = 1000000, beta_dist_a = 0.1, 
                         beta_dist_b = 0.5, bimod_low_lim = -3, 
                         bimod_w_a = 0.5, bimod_w_b = 0.5, bimod_d = 0,
                         bimod_mean_a = 0, trnorm_std_ratio = 2):
        
        # General params
        self.N_wd = N_wd
        self.wd_mean = wd_mean
        
        # For truncated symmetric normals
        self.wd_std = trnorm_std_ratio*wd_mean 
        
        # For Beta distributions
        self.beta_dist_a = beta_dist_a
        self.beta_dist_b = beta_dist_a
        
        # For Bimodal Symmetric Truncated Normals
        self.bimod_low_lim = bimod_low_lim
        
        self.bimod_w_a = bimod_w_a
        self.bimod_w_b = bimod_w_b
        
        self.bimod_d = bimod_d       
        
        self.bimod_mean_a = bimod_mean_a
        
        w_a = bimod_w_a
        w_b = bimod_w_b
        self.bimod_mean_b = (((w_a + w_b)*wd_mean)-(w_a*self.bimod_mean_a))/w_b         
        
        return()	
    #--------------------------------------------------------------------------    


    #--------------------------------------------------------------------------	
    #	Function select distribution of interest
    #--------------------------------------------------------------------------        
    def get_dist_optim_func(self, wd_mean, N_wd = 1000000, dist_sel = 'trunc_norm_sym', manual_shaping = False):
                        
        if(dist_sel == 'trunc_norm_sym'):
            
            if(manual_shaping == False):
                self.set_shape_params(wd_mean, N_wd)
            
            self.optim_func = self.optim_samp_trunc_norm_wd
            
            
        elif(dist_sel == 'beta'):
            
            if(manual_shaping == False):
                self.set_shape_params(wd_mean, N_wd, beta_dist_a = 0.1, 
                                      beta_dist_b = 0.5)
                
            self.optim_func = self.optim_samp_beta_wd
            
            
        elif(dist_sel == 'bimodal'):
            
            if(manual_shaping == False):
                self.set_shape_params(wd_mean, N_wd, bimod_low_lim = -3, 
                                      bimod_w_a = 0.5, bimod_w_b = 0.5, bimod_d = 0,
                                      bimod_mean_a = 0)       
                
            self.optim_func = self.optim_samp_bimodal_sym_wd

        return(self.optim_func)	
    #--------------------------------------------------------------------------    


    #--------------------------------------------------------------------------	
    #	Function to run optimization to generate wd distribution that obeys
    # Jarzynski's equality
    #--------------------------------------------------------------------------        
    def optimize_dist_func(self, wd_mean, N_wd = 1000000, x0 = 2, dist_sel = 'trunc_norm_sym', manual_shaping = False):
                     
        self.get_dist_optim_func(wd_mean, N_wd = N_wd, dist_sel = dist_sel, manual_shaping = manual_shaping)
        
        
        res = minimize(self.optim_func, x0, 
                       method='nelder-mead', 
                       options={'tol': 1e-24,'maxiter': 5000,
                                'disp': True}) 
        self.optim_var = res.x
        
        return(res)	
    #--------------------------------------------------------------------------    


    #--------------------------------------------------------------------------	
    #	Function to run optimization to generate wd distribution that obeys
    # Jarzynski's equality
    #--------------------------------------------------------------------------        
    def get_wd_dist(self, wd_mean, N_samp, dist_sel = 'trunc_norm_sym', x0 = 2, N_wd = 1000000, manual_shaping = False, bypass_optim = False):
                     
        N_samp = int(N_samp)
        N_wd = int(N_wd)
        
        if(bypass_optim == False):
            res = self.optimize_dist_func(wd_mean, N_wd = N_wd, x0 = x0, dist_sel = dist_sel, manual_shaping = manual_shaping)
        
        if(dist_sel == 'trunc_norm_sym'):
            if(manual_shaping == False):
                self.set_shape_params(wd_mean, N_wd)
                
            # Truncated Normal
            wd_low = self.optim_var
            wd_high= 2*wd_mean - wd_low
            wd_samp = self.get_truncated_normal(wd_low, wd_high, wd_mean, self.wd_std, size = N_samp)
        
        elif(dist_sel == 'beta'):
            # Beta Distribution 
            if(manual_shaping == False):
                self.set_shape_params(wd_mean, N_wd, beta_dist_a = 0.1, 
                                      beta_dist_b = 0.5)
                
            a = self.beta_dist_a
            b = self.beta_dist_b
                
            wd_low = self.optim_var
            wd_samp = scbeta.rvs(a, b, size = N_samp)     
            wd_high = ((a+b)/a)*(wd_mean - wd_low) + wd_low
            wd_samp = wd_samp*(wd_high - wd_low) + wd_low
        
        elif(dist_sel == 'bimodal'):
            # Bimodal Distribution 
            if(manual_shaping == False):
                self.set_shape_params(wd_mean, N_wd, bimod_low_lim = -3, 
                                      bimod_w_a = 0.5, bimod_w_b = 0.5, bimod_d = 0,
                                      bimod_mean_a = 0)              
            
            std_a = self.optim_var
            std_b = self.optim_var
            
            wd_samp = self.get_bimodal_sym_trunc_normal(self.bimod_low_lim, self.bimod_mean_a, self.bimod_mean_b, 
                                                   std_a, std_b, w_a = self.bimod_w_a, w_b = self.bimod_w_b, 
                                                   d = self.bimod_d, size = N_samp)        
        
        return(wd_samp)	
    #--------------------------------------------------------------------------    




# =============================================================================    