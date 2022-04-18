# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 11:09:44 2021

Authors
Cailong Hua (hua00023@umn.edu)
Sivaraman Rajaganapathy (sivrmn@umn.edu)

Warning: For research use only
"""

# =============================================================================
# Description
# =============================================================================

# Code for the empirical bounds on the free energy difference given a vector of work don

# =============================================================================
 
 
# =============================================================================
# Imports
# ============================================================================= 
import os
import numpy as np
import matplotlib.pyplot as plt

import scipy as sc
import scipy.special as scp
import scipy.interpolate as sci

import sys
sys.path.append('../../')

# =============================================================================
#%%
# =============================================================================
# Class containing functions for dynamic simulations
# =============================================================================
class NE_Error_Bounds():
    
    #--------------------------------------------------------------------------        
    # Init
    # **T:** Temperature in Kelvin
    #--------------------------------------------------------------------------  
    def __init__(self, T): 
        
        self.T = T
        
        self.kb = sc.constants.Boltzmann
        
        self.beta = 1/(self.kb*self.T)
        
        self.DKWM_bins = 100
    #--------------------------------------------------------------------------    
    
    
    #--------------------------------------------------------------------------
    # Mean and Standard Deviation using log sum exponentials for numerical stability
    #--------------------------------------------------------------------------
    def get_logexp_moments(self, x):
        
        N = np.size(x)
        
        # Mean
        mu_ln = scp.logsumexp(x) - np.log(N)

        # Standard deviation
        e = 2*x 
        f = (2*mu_ln)*np.ones(np.shape(x))
        tot_dat = np.array([e, f]).flatten() 
        
        e_sign = 1*np.ones(np.shape(x))
        f_sign = -1*np.ones(np.shape(x))
        tot_sign = np.array([e_sign, f_sign]).flatten()
                
        std_ln = 0.5*(scp.logsumexp(tot_dat, b = tot_sign) - np.log(N))

        return(mu_ln, std_ln)
    #--------------------------------------------------------------------------
    
    
    #--------------------------------------------------------------------------
    # Empirical Chebyshev Bounds
    # **Arguments:**
    # **exponentiated:** False or True
    #--------------------------------------------------------------------------
    def get_Chebyshev_bound(self, x, delta, exponentiated = False):
        
        
        N = np.size(x)
        
        if(exponentiated == False):
            mu = np.mean(x)
            var = np.var(x)*N/(N-1)      
            
        elif(exponentiated == True):
            # Assumes you want bounds on exp(x)
            [mu_ln, std_ln] = self.get_logexp_moments(x)
            mu = np.exp(mu_ln)
            var = np.exp(2*std_ln)*(N/(N-1))
        else:
            print('Give valid options!')    
    
        # From Ata Kaban
        k = np.sqrt( (1/(delta -(1/N))) * (var/(mu**2)) * ((N**2 - 1))/(N**2))
        
        x_low = mu - k*np.abs(mu)
        x_hi = mu + k*np.abs(mu)
        
        return(x_low, x_hi)
    #--------------------------------------------------------------------------
    
    
    
    #--------------------------------------------------------------------------
    # Empirical Bernstein Bounds
    # **Arguments:**
    # **exponentiated:** False or True
    #--------------------------------------------------------------------------
    def get_Bernstein_bound(self, x, R, delta, exponentiated = False):
        
        N = np.size(x)
        d = delta
        
        if(exponentiated == False):
            x_std = np.std(x)   
            
        elif(exponentiated == True):
            # Assumes you want the bounds on exp(x)
            [mu_ln, std_ln] = self.get_logexp_moments(x)
            x_std = np.exp(std_ln)     
                
        else:
            print('Give valid options!')
            
        # Get Bernstein bound  
        a = x_std * np.sqrt( (2*np.log(3/d)) / N )     
        b = (3*R * np.log(3/d))/N
        
        bern_bnd = a + b
        
        return(bern_bnd)
    #--------------------------------------------------------------------------    
    
    #--------------------------------------------------------------------------
    # Get range on the random variable
    # **Arguments:**
    # **R:** Range on the random variable, enter number if known
    # **exponentiated:** False or True
    #--------------------------------------------------------------------------
    def get_range(self, x, delta_che, R = 'Unknown', exponentiated = False):
        
        if(type(R) == str  and R == 'Unknown'):            
            [x_low, x_hi] = self.get_Chebyshev_bound(x, delta_che, exponentiated = exponentiated)
            R = np.abs(x_hi - x_low)            
        else:
            R = R
        
        return(R)
    #--------------------------------------------------------------------------
    
    
    
    #--------------------------------------------------------------------------
    # Empirical Bounds on random variables with finite support
    # Empirical Bernstein plus an optional Empirical Chebyshev bound on the range. 
    # 
    # **Arguments:**
    # **R:** Range on the random variable, enter number if known
    # **exponentiated:** False or True
    # **confidence_level:** Percent confidence (0-100%)
    # **Returns:**
    # **bound:** Confidance bound of at least 1-delta on the error between the sample mean and the true mean of x
    #--------------------------------------------------------------------------
    def get_emp_bound(self, x, confidence_level, R = 'Unknown', exponentiated = False):
    
        # !!!!! TO RESOLVE: Floating point errors
        delta = 1 - confidence_level/100
    
        if(type(R) == str  and R == 'Unknown'):  
            
            # Sub-optimal way of splitting the confidence deltas between Chebyshev and Bernstein
            # Solving the quadratic by assuming delta_che = delta_bern = d
            # d^2 - 2d + delta = 0 gives d = 1+/- sqrt(1-delta)
            delta_che = 1 - np.sqrt(1-delta)
            delta_bern = delta_che
            
            [x_low, x_hi] = self.get_Chebyshev_bound(x, delta_che, exponentiated = exponentiated)
            R_final = np.abs(x_hi - x_low)    
            
        else:
            R_final = R
            delta_bern = delta
            
            
        bound = self.get_Bernstein_bound(x, R_final, delta_bern, exponentiated = exponentiated)
            
    
        return(bound)
    #--------------------------------------------------------------------------
    
    
    
    
    #--------------------------------------------------------------------------
    # Get the confidence interval on delta_F
    # get_del_F_bounds is changing the bound on exponential term mean(exp(-beta*w)) 
    # into a bound on delta_F
    # **Arguments:**
    # **moments_choice:** 'Stable' or 'Standard'
    #--------------------------------------------------------------------------
    def get_del_F_bounds(self, w, bnd, moments_choice = 'Stable'): 
        
        B = bnd
        
        if(moments_choice == 'Standard'):
            x = np.exp(-self.beta*w)
            A = np.mean(x)
        
        elif(moments_choice == 'Stable'):
            [mu_ln, std_ln] = self.get_logexp_moments(-self.beta*w)
            A = np.exp(mu_ln)     
                        
        else:
            print('Give valid options!')
        
        
        if B >= A:
            up_bound = np.log(A+B)
            low_bound = -np.inf

            # del_F_up_bound = np.nan
            del_F_up_bound = (-low_bound)/(self.beta)
            del_F_low_bound = (-up_bound)/(self.beta)
            
        else:
            up_bound = np.log(A+B)
            low_bound = np.log(A-B)
            del_F_up_bound = (-low_bound)/(self.beta)
            del_F_low_bound = (-up_bound)/(self.beta)
            
        return (del_F_low_bound, del_F_up_bound)
    #--------------------------------------------------------------------------
    
    
    #--------------------------------------------------------------------------
    # Find the CDF from the data set
    # By default the pdf and cdf are left semi-continuous 
    # bin_ranges are [b_low, b_high)
    # **Arguments:**
    # 
    #--------------------------------------------------------------------------    
    def get_CDF(self, x, bins = None, x_range = None):
    
        if(bins != None):
            [x_pdf, bin_edges] = np.histogram(x, bins = bins, range = x_range, density = True)    
        else:
            
            bin_edges= np.unique(x)
            if(np.size(bin_edges) < 100):
                [x_pdf, bin_edges] = np.histogram(x, bins = 100, range = x_range, density = True)
            
            else:
                [x_pdf, bin_edges] = np.histogram(x, bins = bin_edges, range = x_range, density = True)    
    
            
        x_cdf = np.cumsum(x_pdf)
        x_cdf = x_cdf/np.max(x_cdf)
        
        bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])

        return(x_cdf, x_pdf, bin_edges, bin_centers)
    #--------------------------------------------------------------------------
    


    #--------------------------------------------------------------------------
    # Find the DKWM CDF bounds for a given data set with confidence 1-alpha
    # bin_ranges are [b_low, b_high)
    # **Arguments:**
    # 
    #--------------------------------------------------------------------------    
    def get_DKWM_CDF_bounds(self, x, alpha = 0.05, bins = None, diagnostics = True):
    
        N = np.size(x)
        
        [x_cdf, x_pdf, bin_edges, bin_centers] = self.get_CDF(x, bins = bins)
                
        # DKWM bounds
        epsilon = np.sqrt((np.log(2/alpha))/(2*N))
    
        # CDF uncerntaity bounds have to belong to [0,1]
        DKWM_low = np.clip(x_cdf - epsilon, 0, 1)
        
        DKWM_high = np.clip(x_cdf + epsilon, 0, 1)
        

        if(diagnostics == True):
            
            x_cdf_aug = np.hstack([x_cdf, x_cdf[-1]])
            DKWM_low_aug = np.hstack([DKWM_low, DKWM_low[-1]])
            DKWM_high_aug = np.hstack([DKWM_high, DKWM_high[-1]])
            
            plt.figure()
            bin_edges_plot = bin_edges * self.beta
            plt.plot(bin_edges_plot, x_cdf_aug, color = 'b', label = '$\widehat{\Phi}_N$')#CDF_est')
            plt.plot(bin_edges_plot, DKWM_low_aug, color = 'r', label = '$\widehat{\Phi}_N$ - $\epsilon$')#'Lower_CDF')
            plt.plot(bin_edges_plot, DKWM_high_aug, color = 'g', label = '$\widehat{\Phi}_N$ + $\epsilon$' )#'Upper_CDF')

            ax = np.random.choice(bin_edges_plot,1)
            ay = np.interp(ax, bin_edges_plot, x_cdf_aug)
            # plt.annotate(text='', xy=(ax,ay), xytext=(ax,0), arrowprops=dict(arrowstyle='<->'))
            # plt.annotate(text='', xy=(ax,ay), xytext=(ax,1), arrowprops=dict(arrowstyle='<->'))
            # plt.text(ax+0.1, (ay+0)/2, 'd1', rotation = 0)
            # plt.text(ax+0.1, (ay+1)/2, 'd2', rotation = 0)
            # plt.xticks(ax, 'a')
            
            bx = np.random.choice(bin_edges_plot,1)
            by = np.interp(bx, bin_edges_plot, x_cdf_aug)
            # plt.annotate(text='', xy=(bx,by+epsilon), xytext=(bx,by), arrowprops=dict(arrowstyle='<->'))
            # plt.text(bx+0.1, by+epsilon/2, 'e', rotation = 0)
            plt.plot((bx,bx), (by,by+epsilon), linewidth  = 2, color = 'k')
            plt.annotate(text='$\epsilon$', xy=(bx,by+epsilon/2), xytext=(bx+15,by+0.05), arrowprops=dict(arrowstyle='->'))
            
            plt.xlabel('w (kbT)')
            plt.ylabel('CDF(w)')
            plt.legend()
            plt.title('Confidence Level = ' + str(1 - round(alpha,2)) + '| N = ' + str(N))   
            
            
        
        return(DKWM_low, DKWM_high, x_cdf, bin_edges)
    #--------------------------------------------------------------------------


    #--------------------------------------------------------------------------
    # Find the value of the cdf at a give point
    # **Arguments:**
    # 
    #--------------------------------------------------------------------------    
    def get_CDF_interpol(self, cdf_vals, bin_edges):
    
            
        mod_cdf_vals = np.hstack([cdf_vals[0], cdf_vals, cdf_vals[-1], cdf_vals[-1]])
        mod_bin_edges = np.hstack([-np.inf, bin_edges, np.inf])

        cdf_func = sci.interp1d(mod_bin_edges, mod_cdf_vals, kind = 'zero')
        
        return(cdf_func)
    #--------------------------------------------------------------------------




    #--------------------------------------------------------------------------
    # Find the 'H' function
    # **Arguments:**
    # 
    #--------------------------------------------------------------------------    
    def get_h_func_legacy(self, zeta, w_max, cdf_a_vals, cdf_b_vals, bin_edges):
    
        # Pick from lower bound of DKWM for h*
        cdf_a_func = self.get_CDF_interpol(cdf_a_vals, bin_edges)
        gamma_a = cdf_a_func(zeta)
        
        # Pick from upper bound of DKWM for h*
        cdf_b_func = self.get_CDF_interpol(cdf_b_vals, bin_edges)
        gamma_b = cdf_b_func(zeta)
        
        a = gamma_a*np.exp(-self.beta*zeta) 
        b = (1-gamma_b)*np.exp(-self.beta*w_max)
        
        # original
        h_vals = (-1/self.beta)*np.log(a + b)

        return(h_vals)
    #--------------------------------------------------------------------------


    #--------------------------------------------------------------------------
    # Find the 'G' function
    # **Arguments:**
    # 
    #--------------------------------------------------------------------------    
    def get_g_func_legacy(self, zeta, w_min, cdf_a_vals, cdf_b_vals, bin_edges):
    
        # Pick from upper bound of DKWM for g*
        cdf_a_func = self.get_CDF_interpol(cdf_a_vals, bin_edges)
        gamma_a = cdf_a_func(zeta)
        
        # Pick from lower bound of DKWM for g*
        cdf_b_func = self.get_CDF_interpol(cdf_b_vals, bin_edges)
        gamma_b = cdf_b_func(zeta)
        
        a = gamma_a*np.exp(-self.beta*w_min)
        b = (1-gamma_b)*np.exp(-self.beta*zeta)
        
        g_vals = (-1/self.beta)*np.log(a + b)
        
        return(g_vals)
    #--------------------------------------------------------------------------



    #--------------------------------------------------------------------------
    # Find the 'H' function
    # **Arguments:**
    # 
    #--------------------------------------------------------------------------    
    def get_h_func(self, zeta, w_min, w_max, DKWM_low, DKWM_high, bin_edges):
    
        
        cdf_low_func = self.get_CDF_interpol(DKWM_low, bin_edges)
        cdf_high_func = self.get_CDF_interpol(DKWM_high, bin_edges)

        # Implementing the upper bound function ("U" in the paper)    
        cdf_low_zeta = cdf_low_func(zeta)
        cdf_high_w_min = cdf_high_func(w_min)
        term_1 = np.exp(-self.beta*zeta)*(cdf_low_zeta - cdf_high_w_min)
        
        cdf_low_w_max = cdf_low_func(w_max)
        cdf_high_zeta = cdf_high_func(zeta)
        term_2 = np.exp(-self.beta*w_max)*(cdf_low_w_max - cdf_high_zeta)
        
        U = term_1 + term_2
        
        # Check if U is 0 or negative. In this case, make h_vals = w_max
        U_neg_chk = (U <= 0)
        U[U_neg_chk] = np.exp(-self.beta*w_max)
        
        h_vals = (-1/self.beta)*np.log(U)
        
        return(h_vals)
    #--------------------------------------------------------------------------




    #--------------------------------------------------------------------------
    # Find the 'G' function
    # **Arguments:**
    # 
    #--------------------------------------------------------------------------    
    def get_g_func(self, zeta, w_min, w_max, DKWM_low, DKWM_high, bin_edges):
    
        cdf_low_func = self.get_CDF_interpol(DKWM_low, bin_edges)
        cdf_high_func = self.get_CDF_interpol(DKWM_high, bin_edges)

        # Implementing the upper bound function ("U" in the paper)    
        cdf_low_w_min = cdf_low_func(w_min)
        cdf_high_zeta = cdf_high_func(zeta)
        term_1 = np.exp(-self.beta*w_min)*(cdf_high_zeta - cdf_low_w_min)
        
        cdf_low_zeta = cdf_low_func(zeta)
        cdf_high_w_max = cdf_high_func(w_max)
        term_2 = np.exp(-self.beta*zeta)*(cdf_high_w_max - cdf_low_zeta)
        
        L = term_1 + term_2
        
        g_vals = (-1/self.beta)*np.log(L)
        
        return(g_vals)
    #--------------------------------------------------------------------------



    #--------------------------------------------------------------------------
    # Find GH bounds
    # **Arguments:**
    # 
    # N - For the resolution of zeta over which to optimize the bound functions h* & g*
    #--------------------------------------------------------------------------    
    def get_gh_bounds(self, w, w_min, w_max, alpha = 0.05, N = 100, bins = None, diagnostics = False, legacy = False):
    
        # 1. Find the cdf of the work distribution & the confidence bounds on that
        [DKWM_low, DKWM_high, x_cdf, bin_edges] = self.get_DKWM_CDF_bounds(w, alpha = alpha, bins = bins, diagnostics = diagnostics)
        
        # 2. Define the range of work for which to compute the bounds
        zeta = np.linspace(w_min, w_max, num = N)
        
        # 3. Compute the upper bound function H*
        if(legacy == True):
            h_star = self.get_h_func_legacy(zeta, w_max, DKWM_low, DKWM_high, bin_edges)
        else:
            h_star = self.get_h_func(zeta, w_min, w_max, DKWM_low, DKWM_high, bin_edges)
                
        # 4. Compute the lower bound function G*
        if(legacy == True):
            g_star = self.get_g_func_legacy(zeta, w_min, DKWM_high, DKWM_low, bin_edges)
        else:
            g_star = self.get_g_func(zeta, w_min, w_max, DKWM_low, DKWM_high, bin_edges)

        
        # 5. Compute the smallest upper bound U, and the largest lower bound L
        gh_U = np.min(h_star)
        gh_L = np.max(g_star)
        
        if(diagnostics == True):
            
            plt.figure()
            plt.plot(zeta, h_star, color = 'b')
            plt.plot(zeta, g_star, color = 'g')
            plt.plot(zeta, w_min*np.ones(np.shape(zeta)), color = 'k')
            plt.plot(zeta, w_max*np.ones(np.shape(zeta)), color = 'r')
            
            plt.title('G_star and H_star functions')
            plt.xlabel('Work (zeta)')
            plt.legend(['H_star', 'G_star', 'W_min', 'W_max'])
        
        return(gh_L, gh_U, g_star, h_star, zeta)
    #--------------------------------------------------------------------------



    #--------------------------------------------------------------------------
    # Get bounds using Chebyshev-Bernstein or Chebyshev-DWKM
    # The units of the bounds will be the same as the units of the work & range values
    #--------------------------------------------------------------------------
    def get_neq_error_bounds(self, w, conf_level, R = 'Unknown', bound_type = 'Combined', work_in_joules = False):
        """
        
        Parameters
        ----------
        **w** : TYPE Numpy array
            Array of non-equlibrium work
        conf_level : TYPE
            DESCRIPTION.
        R : TYPE, optional
            DESCRIPTION. The default is 'Unknown'.
        bound_type : TYPE, optional
            DESCRIPTION. The default is 'Combined'.
        work_in_joules : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        del_F_l_bnd: TYPE float
            Lower bound on dF = \[e^(-$\beta$ w)\] 
 
        """
        # Take work values in joules or kbT and generate bounds on dF
        if(bound_type == 'Combined'):
            conf_level = np.sqrt(conf_level/100)*100
        
        
        if((bound_type == 'CB') or (bound_type == 'Combined')):
            # Chebyshev-Bernstein Bounds        
            # Work in joules
            if(work_in_joules == True):
                # Bound on |exp(-beta*w) -  exp(-beta*dF)|
                bern_bound = self.get_emp_bound(-self.beta*w, conf_level, R = R, exponentiated = True)
                [CB_del_F_l_bnd, CB_del_F_u_bnd] = self.get_del_F_bounds(w, bern_bound)
            # Work in kbT
            else:
                # Bound on |exp(-beta*w) -  exp(-beta*dF)|
                bern_bound = self.get_emp_bound(-w, conf_level, R = R, exponentiated = True)
                [CB_del_F_l_bnd, CB_del_F_u_bnd] = self.get_del_F_bounds(w/self.beta, bern_bound)
            
            if(work_in_joules == False):
                CB_del_F_l_bnd = self.beta*CB_del_F_l_bnd
                CB_del_F_u_bnd = self.beta*CB_del_F_u_bnd
                
                         
            
        if((bound_type == 'DKWM') or (bound_type == 'Combined')):   
            # DKWM Bounds
            delta = 1 - conf_level/100
            # Use Chebyshev to get the bound on w
            if(type(R) == str  and R == 'Unknown'):    
                delta_che = 1 - np.sqrt(1-delta)
                delta_DWKM = delta_che
                [w_min,w_max] = self.get_Chebyshev_bound(w, delta_che, exponentiated = False)
            
            else:
                delta_DWKM = delta
                w_min = R[0]
                w_max = R[1]
                
            if(work_in_joules == True):
                [gh_L, gh_U, g_star, h_star, zeta] = self.get_gh_bounds(w, w_min, w_max, alpha = delta_DWKM, N = 100, bins = self.DKWM_bins, diagnostics = False, legacy = False)
            else:
                [gh_L, gh_U, g_star, h_star, zeta] = self.get_gh_bounds(w/self.beta, w_min/self.beta, w_max/self.beta, alpha = delta_DWKM, N = 100, bins = None, diagnostics = False, legacy = False)
                gh_L = gh_L*self.beta
                gh_U = gh_U*self.beta        
                
                
        if(bound_type == 'CB'):        
            del_F_l_bnd = CB_del_F_l_bnd
            del_F_u_bnd = CB_del_F_u_bnd                  
                    
        elif(bound_type == 'DKWM'):
            del_F_l_bnd = gh_L
            del_F_u_bnd = gh_U   
            
        elif(bound_type == 'Combined'):
            del_F_l_bnd = np.max([CB_del_F_l_bnd, gh_L])
            del_F_u_bnd = np.min([CB_del_F_u_bnd, gh_U])

        
        return(del_F_l_bnd, del_F_u_bnd)
    #--------------------------------------------------------------------------

    
# =============================================================================





    