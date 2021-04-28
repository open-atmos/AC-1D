1D model for ice formation from INP
==============================================

The goal of this simplified 1D model is to enable the comparison and evaluation of different ice formation mechanisms from the literature against observational constraints. The model is informed from LES case study output.

Assumptions
^^^^^^^^^^^^^^

1. Cloud conditions are assumed steady-state (unaffected by weak ice formation).  
2. All aerosol are assumed to be activated in updrafts and restored in downdrafts.  
3. Thus, all in-cloud aerosol (including INP) are within a droplet suitable for immersion freezing.  

Step-by-step (model operation)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1. Profiles of temperature, RH, and cloud water mixing ratio are read in from LES output.  
2. Model predicts evolution of Ninp(z,t) and Nice(z,t).  
3. Precipitation rates are estimated from LES number-weighted value at cloud base.  
4. Predicted nucleation rate profiles are also saved and plotted.  

Requirements
^^^^^^^^^^^^

* Numpy (https://numpy.org)
* Matplotlib (https://matplotlib.org)
* Xarray (http://xarray.pydata.org)
   
Authors
-------

Code was written by Ann Fridlind (NASA GISS). 

References
----------

