
<p float="center">
  <img src="Figures/Logo/NET_Crop.svg" title = "Non-Equilibrium Toolbox" width = "1000"  />
</p>

**Non-Equilibrium Toolbox (NET):** For computing confidence bounds on the free-energy estimates computed from non-equilibrium experiments via the Jarzynski equality. 

# How to use
Here is a sample code to compute the confidence bounds on the Jarzynski estimator in 5 easy steps:

```python
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

```

# Examples
A summary of the performance and characteristics of the bounds on the synthetic data. (a) A detailed view of the performance results. The
box plot shows (left) showcases the performance of the Combined Bounds
at $95\%$ confidence level on work distributions that are truncated
normals, with $\left\langle w_{d}\right\rangle /\Delta F=0.5$ as
the number of samples are increased. Each box plot represents one
case of the synthetic data validation, wherein each case comprises
of a collection of $500$ computations of the bounds. An example case
$\left(N=3034\right)$ is shown in more detail on the right. The summarized
view of the bounds from the synthetic data study, with varying distributions,
varying $\left\langle w_{d}\right\rangle $, and increasing $N$,
is shown in (b). The performance curves have been obtained using $500$
computations for each case and interpolating the modes of the data
obtained in each case.

<p float="left">
  <img src="Figures/Synthetic_data_results_details.png" title = "Synthesized data result details" width = "1000"  />
  <img src="Figures/Synthetic_data_results.png" title = "Synthesized data result" width = "1000" /> 
</p>



# Citation 
To cite this toolbox, please use:

- Rajaganapathy, Sivaraman, and Murti Salapaka. "Confidence bounds for the Jarzynski estimator." Bulletin of the American Physical Society (2022).
