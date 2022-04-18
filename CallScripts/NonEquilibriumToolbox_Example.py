# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 14:35:37 2022

@author: siva_
"""

import numpy as np
import scipy.special as scp

# STEP 1: Import toolbox
import sys
sys.path.append('../')

from APIs.NE_ErrorBnds.NE_Error_Bounds import NE_Error_Bounds
from APIs.Models.Random_Data_Generator import Random_Data_Generator

# STEP 2: Set experimental parameters & desired confidence levels 
T = 273 + 25 # Experimental temperature in Kelvin
conf_level = 95 # Confidence Level on the Bounds in Percentage

# STEP 3: Initialize the toolbox
RDGObj = Random_Data_Generator(T) 
NEBObj = NE_Error_Bounds(T)

# STEP 4: Generate work samples (replace STEP 4 with actual data)
N_samp = 1e3 # No. of work samples
wd_mean = 10 # Measured in KbT 
dF_true = 20 # Ground truth Free Energy Change in KbT
wd = RDGObj.get_wd_dist(wd_mean, N_samp) # Generating dissipation work distribution 
w = wd + dF_true # Finding the work distribution

# STEP 5: Compute bounds on dF & an estimate of dF
[dF_lower_bound, dF_upper_bound] = NEBObj.get_neq_error_bounds(w, conf_level) # Bounds

dF_estimate = NEBObj.get_Jarzynski_estimate(w) # dF estimate

print('Free Energy Difference = ' + str(round(dF_true,2)) + ' KbT')
print('The Jarzynski Estimate = ' + str(round(dF_estimate,2))+' KbT')
print('Lower Bound on estimates at '+str(round(conf_level,2))+ '% Confidence = ' + str(round(dF_lower_bound,2))+ ' KbT')
print('Upper Bound on estimates at '+str(round(conf_level,2))+ '% Confidence = ' + str(round(dF_upper_bound,2))+ ' KbT')

