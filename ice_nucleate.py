"""
This file is used to nucleate ice using the ABIFM or singular approaches.
"""
import xarray as xr
import numpy as np
import pandas as pd


class Jhet():
    """
    Class to load Jhet LUT and assign c and m coefficient values based on requested INP type.
    """
    def __init__(self, inp_type="Illite", coeff_filename="Jhet_coeff.csv"):        
        """
        set the ABIFM linear fit coefficients for the INP type.

        Parameters
        ---------
        inp_type: str
            INP type to use (must match the LUT (not case sensitive).
        coeff_filename: str
            path and filename of Jhet coefficients' Table. By default using the values from Table 1 in Knopf and
            Alpert, 2013, DOI: 10.1039/C3FD00035D.
        """
        self.jhet_coeff = self._load_jhet_coeff(coeff_filename=coeff_filename)
        self._set_jhet_coeff(inp_type=inp_type)

    def _load_jhet_coeff(self, coeff_filename="Jhet_coeff.csv"):
        """
        Loads Jhet coefficients tables assuming that the columns represent (from left to right): INP type
        (substance), c coefficient, c SD, lower and upper confidence levels for c (respectively), m coefficient,
        m SD, and lower and upper confidence levels for m.

        Parameters
        ---------
        coeff_filename: str
            path and filename of Jhet coefficients' Table. By default using the values from Table 1 in Knopf and
            Alpert, 2013, DOI: 10.1039/C3FD00035D.

        Returns
        -------
        jhet_coeff: DataFrame
        The Jhet coefficients including c (slope) and m (intercept) required for the Jhet calculation.

        """
        jhet_coeff = pd.read_csv(coeff_filename, names=["inp_type", "c", "sigma_c", "LCL_c", "UCL_c", "m",
                                                        "sigma_m", "LCL_m", "UCL_m"], index_col=0)
        return jhet_coeff

    def _set_jhet_coeff(self, inp_type="Illite"):
        """
        set the ABIFM linear fit coefficients for the specified INP type.
        """
        if inp_type.lower() in self.jhet_coeff.index.str.lower():  # allowing case errors
            subs_loc = self.jhet_coeff.index.str.lower() == inp_type.lower()
            self.c, self.m = np.float64(self.jhet_coeff.loc[subs_loc, ["c", "m"]].values)[0]
        else:
            raise NameError("INP type '%s' not found in Jhet table" % inp_type)


def use_abifm(model, inp_init=2.e0, inp_diam=1.e-4, w_e_ent=0.1e-3, tau_mix=1800., v_f_ice=0.3):
    """
    Nucleate ice using the a_w based immersion freezing model (ABIFM)

    Parameters
    ---------
    inp_init: float
        INP number concentration [L-1].
    inp_diam: float
        particle diameter
    w_e_ent: float
        cloud-top entrainment rate [m/s]. default value derived from LES offline (see Fridlind et al. 2012,
        doi:10.1175/JAS-D-11-052.1)
    tau_mix: float
        boundary-layer mixing time scale [s]. default value derived from LES offline (see Fridlind et al. 2012,
        doi:10.1175/JAS-D-11-052.1)
    v_f_ice: float
        number-weighted ice crystal fall speed at the surface [m/s]. default value derived from LES offline
        (see Fridlind et al. 2012, doi:10.1175/JAS-D-11-052.1)

    Output
    ---------
    model: Xarray DataSet
        Dataset containing all model fields
    """

    # domain-uniform initial monomodal INP that are also hygroscopic
    inp_surf = 4*np.pi*(inp_diam/2.)**2   # surface area per particle [cm2]

    # Calculate Jhet
    Jhet = 10.**(c + m * les.delta_aw.values)    # nucleation rate [cm-2 s-1]

