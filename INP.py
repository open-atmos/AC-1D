"""
This module include INP population class and its sub-classes for different PSDs.
In addition, it includes the Jhet class.
"""
import xarray as xr
import numpy as np
import pandas as pd


class Jhet():
    """
    Class to load Jhet LUT and assign c and m coefficient values based on requested INP type.
    """
    def __init__(self, nucleus_type="Illite", coeff_filename="Jhet_coeff.csv"):
        """
        set the ABIFM linear fit coefficients for the INP type.

        Parameters
        ---------
        nucleus_type: str
            INP type to use (must match the LUT (not case sensitive).
        coeff_filename: str
            path and filename of Jhet coefficients' Table. By default using the values from Table 1 in Knopf and
            Alpert, 2013, DOI: 10.1039/C3FD00035D.
        """
        self.Jhet_coeff_table = self._load_Jhet_coeff(coeff_filename=coeff_filename)
        self._set_Jhet_coeff(nucleus_type=nucleus_type)

    def _load_Jhet_coeff(self, coeff_filename="Jhet_coeff.csv"):
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
        Jhet_coeff_table: DataFrame
        The Jhet coefficients including c (slope) and m (intercept) required for the Jhet calculation.

        """
        Jhet_coeff_table = pd.read_csv(coeff_filename, names=["nucleus_type", "c", "sigma_c", "LCL_c", "UCL_c",
                                                              "m", "sigma_m", "LCL_m", "UCL_m"], index_col=0)
        return Jhet_coeff_table

    def _set_Jhet_coeff(self, nucleus_type="Illite"):
        """
        set the ABIFM linear fit coefficients for the specified INP type.
        """
        if nucleus_type.lower() in self.Jhet_coeff_table.index.str.lower():  # allowing case errors
            subs_loc = self.Jhet_coeff_table.index.str.lower() == nucleus_type.lower()
            self.c, self.m = np.float64(self.Jhet_coeff_table.loc[subs_loc, ["c", "m"]].values)[0]
        else:
            raise NameError("INP type '%s' not found in Jhet table" % nucleus_type)


class INP_pop():
    """
    class for INP population
    """
    def __init__(self, use_ABIFM=None, n_init_max=None, nucleus_type=None, diam=None, dn_dlogD=None, name=None,
                 diam_cutoff=None, T_array=None, singular_fun=None, singular_scale=None, psd={},
                 n_init_weight_prof=None, ci_model=None):
        """
        INP population namelist

        Parameters and Attributes
        -------------------------
        use_ABIFM: bool
            True - use ABIFM, False - use singular.
        scheme: str
            "ABIFM" or "singular" or None when no input is provided.
        nucleus_type: str
            type of nucleus required for Jhet calculation (--ABIFM--)
        Jhet: Jhet class object
            To use with ABIFM (--ABIFM--)
        diam_cutff: float or tuple
            If float, minimum particle diameter to consider (0.5 by default, i.e., consider all sizes).
            If tuple, then lower and upper diameter limits (--singular--).
        T_array: list or ndarray
            discrete temperature [K] array for INP parametrization (--singular--).
        singular_fun: str or lambda function
            If str, then:
                1. use "D2010" to use eq. 1 in DeMott et al., 2010.
                2. "D2015" to use eq. 2 in DeMott et al..
                3. "D2010fit" to use the temperature dependence fit from fig. 2 caption in DeMott et al., 2010.
                4. "ND20212 to use surface area temperature-based fit (eq. 5) in Niemand et al., JAS, 2012.
                Use a lambda function for INP parametrization typically as a function of T (--singular--).
            Use "D2010" (default) if None.
            Notes:
            The D2015 has default values of the five coefficients from eq. 2 (cf - calibration correction factor,
            alpha, beta, gamma, delta); these might be coded as optional input parameters for INP the class in
            the future.
            "D2010fit" does not consider aerosol PSDs.
        singular_scale: float
            Scale factor for 'singular_fun' (1 by default) (--singular--).
        n_init_max: float
            total initial INP concentration [L-1].
        diam: list or ndarray or scalar
            discrete particle diameter array [um]
        dn_dlogD: list or ndarray or scalar
            discrete particle number per size bin (sums to n_init_max) [L-1]
        psd_type: str
            population type e.g., "mono", "logn", "multi_logn", "custom".
        psd: dict
            dictionary providing psd parameter information enabling full psd reproduction.
        name: str
            population name (or tag).
        n_init_weight_prof: dict or None
               a dict with keys "height" and "weight". Each key contains a list or np.ndarray of length s (s > 1)
               determining PSD heights [m] and weighting profiles. Weights are applied on n_init such that
               n_init(z) = n_init_max * weighting_factor(z), i.e., a weighted_inp_prof filled with ones means
               that n_init(z) = n_init_max.
               Weights are generally expected to have values between 0 and 1. If at least one weight value > 1,
               then the profile is normalized such that the maximum value equals 1. heights are interpolated
               between the specified heights, and the edge values are used for extrapolation (can be used to set
               different INP source layers at model initialization, and combined with turbulence weighting,
               allows the emulation of cloud-driven mixing.
        ci_model: ci_model class
            Containing variables such as the requested domain size, LES time averaging option
            (ci_model.t_averaged_les), custom or LES grid information (ci_model.custom_vert_grid),
            and LES xr.DataSet object(ci_model.les) after being processed.
            All these required data are automatically set when a ci_model class object is assigned
            during model initialization.
        ds: Xarray Dataset
            will be shaped and incorporate all population INP in the domain:
            ABIFM: height x time x diameter
            singular: height x time x T
        """
        # set attributes
        if use_ABIFM is not None:
            if not np.logical_or(not use_ABIFM,
                                 np.logical_and(nucleus_type is not None, diam is not None)):
                raise RuntimeError("nucleus type and diameter must be specified in ABIFM")
            if use_ABIFM:
                self.scheme = "ABIFM"
                self.nucleus_type = nucleus_type
                self.Jhet = Jhet(nucleus_type=nucleus_type)
            else:
                self.scheme = "singular"
                if diam_cutoff is None:
                    self.diam_cutoff = 0.5
                else:
                    self.diam_cutoff = diam_cutoff
                if singular_fun is None:
                    singular_fun = "D2010"  # set by default to DeMott et al. (2010)
                if singular_scale is None:
                    self.singular_scale = 1.
                else:
                    self.singular_scale = singular_scale
        else:
            self.scheme = None
        self.n_init_weight_prof = n_init_weight_prof
        self.n_init_max = n_init_max
        if isinstance(diam, (float, int)):
            self.diam = [diam]
        else:
            self.diam = diam
        if isinstance(dn_dlogD, (float, int)):
            self.dn_dlogD = [dn_dlogD]
        else:
            self.dn_dlogD = dn_dlogD
        self.psd = psd
        if "type" in psd.keys():  # assuming that the __init__ method is invoked from an inhertied classes.
            self.psd_type = psd["type"]
        if name is None:
            self._random_name()
        else:
            self.name = name

        # Assign INP dataset
        self.ds = xr.Dataset()
        self.ds = self.ds.assign_coords({"diam": np.ones(1) * diam})
        self.ds["dn_dlogD"] = xr.DataArray(np.ones(1) * dn_dlogD, dims=self.ds["diam"].dims)
        self.ds["dn_dlogD"].attrs["units"] = "$L^{-1}$"
        self.ds["dn_dlogD"].attrs["long_name"] = "Particle number concentration per diameter bin"
        self._calc_surf_area()

        # Use the ci_model class object (if provided) to initialize the INP array (start with height-time coords).
        if ci_model is not None:
            self.ds = self.ds.assign_coords({"height": ci_model.ds["height"].values,
                                             "time": ci_model.ds["time"].values})
            if use_ABIFM is True:
                self._init_inp_Jhet_ABIFM_arrays(ci_model)
                if n_init_weight_prof is not None:
                    self._weight_inp_prof()
            elif use_ABIFM is False:
                if T_array is None:  # By default using 1st decimal T to allocate INP array with delta T = 0.1 C
                    if ci_model.ds["T"].min() - 273.15 >= -5:
                        raise RuntimeError('Minimum LES-informed temperature must be larger than -5 C in'\
                            ' singular mode to allow any INP to activate')
                    T_min = np.maximum(np.floor((ci_model.ds["T"].min().values - 273.15) * 10) / 10, -40)
                    self.T_array = np.linspace(T_min, -5., int(np.around((-5 - T_min) * 10)) + 1) + 273.15
                else:
                    self.T_array = T_array
                self._set_inp_conc_fun(singular_fun)
                self._init_inp_singular_array()
            self.ds["height"].attrs["units"] = "$m$"
            self.ds["time"].attrs["units"] = "$s$"
            self.ds["time_h"] = self.ds["time"].copy() / 3600  # add coordinates for time in h.
            self.ds["time_h"].attrs["units"] = "$h$"

        else:
            print("'ci_model' object not provided - not assigning INP array")

        # Set coordinate attributes
        self.ds["diam"].attrs["units"] = r"$\mu m$"
        self.ds["diam"].attrs["long_name"] = "Particle diameter"

    def _random_name(self):
        """
        Generate random string name for population
        """
        self.name = self.psd_type + \
            "_%05d" % np.random.randint(1e4 - 1)  # generate random population number if not provided.

    def _calc_surf_area(self):
        """
        Calculate surface area per particle [cm2] corresponding to each diameter to use with Jhet [cm-2 * s-1]
        """
        self.ds["surf_area"] = xr.DataArray(np.pi * (self.ds["diam"] * 1e-4) ** 2, dims=self.ds["diam"].dims)
        self.ds["surf_area"].attrs["units"] = "$cm^2$"
        self.ds["surf_area"].attrs["long_name"] = "Surface area per particle diameter"

    def _set_inp_conc_fun(self, singular_fun):
        """
        Set the INP initialization function for the singular approach.
        Parameters
        ---------
        singular_fun: str or lambda function
            If str, then:
                1. use "D2010" to use eq. 1 in DeMott et al., 2010.
                2. "D2015" to use eq. 2 in DeMott et al..
                3. "D2010fit" to use the temperature dependence fit from fig. 2 caption in DeMott et al., 2010.
                4. "ND20212 to use surface area temperature-based fit (eq. 5) in Niemand et al., JAS, 2012.
            Use a lambda function for INP parametrization typically as a function of T (--singular--).
            Use "D2010" (default) if None.
            Notes:
            The D2015 has default values of the five coefficients from eq. 2 (cf - calibration correction factor,
            alpha, beta, gamma, delta); these might be coded as optional input parameters for INP the class in
            the future.
            "D2010fit" does not consider aerosol PSDs.
        """
        if isinstance(singular_fun, str):
            if singular_fun == "D2010":
                self.singular_fun = lambda Tk, n_aer05: 0.0000594 * (273.16 - Tk) ** 3.33 * n_aer05 ** \
                    (0.0264 * (273.16 - Tk) + 0.0033)  # DeMott et al. (2010)
            elif singular_fun == "D2015":
                self.singular_fun = \
                    lambda Tk, n_aer05, cf=3., alpha=0., beta=1.25, gamma=0.46, delta=-11.6: \
                    cf * n_aer05 ** (alpha * (273.16 - Tk) + beta) * \
                    np.exp(gamma * (273.16 - Tk) + delta)  # DeMott et al. (2015)
            elif singular_fun == "D2010fit":
                self.singular_fun = \
                    lambda Tk: 0.117 * np.exp(-0.125 * (Tk - 273.2))  # DeMott et al. (2010) fig. 2 fit
            elif singular_fun == "ND2012":
                self.singular_fun = lambda Tk, s_area: \
                    np.exp(-0.517 * (Tk - 273.15) + 8.934) * s_area * 1e-4  # INAS Niemand et al. (2012)
            else:
                raise NameError("The singular treatment %s is not implemented in the model. Check the \
                                input string." % singular_fun)
        else:  # assuming lambda function
            self.singular_fun = singular_fun

    def _init_inp_singular_array(self):
        """
        initialize the INP array for singular (height x time x temperature).
        Parameters
        ---------
        param_dict: dict
            Keys include all possible input parameters for the INP sub-classes.

        Returns
        -------
        tmp_inp_pop: INP class object
            INP class object that includes the INP array with dims height x time x diameter (ABIFM) or
            height x time x temperature (singular).
        """
        self.ds = self.ds.assign_coords({"T": self.T_array})
        tmp_inp_array = np.zeros((self.ds["height"].size, self.ds["T"].size))
        if self.singular_fun.__code__.co_argcount > 1:
            if 'n_aer05' in self.singular_fun.__code__.co_varnames:  # 2nd argument is aerosol conc. above cutoff
                if isinstance(self.diam_cutoff, float):
                    input_2 = np.sum(self.ds["dn_dlogD"].sel({"diam": slice(self.diam_cutoff, None)}).values)
                else:  # assuming 2-element tuple
                    input_2 = np.sum(self.ds["dn_dlogD"].sel({"diam": slice(self.diam_cutoff[0],
                                                                            self.diam_cutoff[1])}).values)
                input_2 = np.ones((self.ds["height"].size, self.ds["T"].size)) * input_2
                tmp_n_inp = np.flip(self.singular_fun(np.tile(np.expand_dims(self.ds["T"].values, axis=0),
                                    (self.ds["height"].size, 1)), input_2), axis=1)  # start at max temperature
            elif 's_area' in self.singular_fun.__code__.co_varnames:  # 2nd argument is surface area
                input_2 = np.sum(self.ds["surf_area"].values)

                # Add initial INP diagnostic field (diameter vs. T) for INAS.
                self.ds["init_inp"] = \
                    xr.DataArray(self.singular_fun(np.tile(np.expand_dims(np.flip(self.ds["T"].values), axis=0),
                                                           (self.ds["diam"].size, 1)),
                                                   np.tile(np.expand_dims(self.ds["surf_area"].values, axis=1),
                                                           (1, self.ds["T"].size))),
                                 dims=("diam", "T")) * self.ds["dn_dlogD"]
                tmp_n_inp = np.tile(np.expand_dims(self.ds["init_inp"].sum("diam").squeeze(), axis=0),
                                    (self.ds["height"].size, 1))
                for ii in range(1, self.ds["T"].size):
                    self.ds["init_inp"][:, ii] = self.ds["init_inp"][:, ii] - self.ds["init_inp"][:, :ii].sum("T")
                self.ds["init_inp"].values = np.flip(self.ds["init_inp"].values, axis=1)

            # weight array vertically.
            if self.n_init_weight_prof is not None:
                tmp_n_inp = np.tile(np.expand_dims(self._weight_inp_prof(False), axis=1),
                                  (1, self.ds["T"].size)) * tmp_n_inp
        else:  # single input (temperature)
            tmp_n_inp = np.tile(np.expand_dims(np.flip(self.singular_fun(self.ds["T"].values)), axis=0),
                                (self.ds["height"].size, 1))  # start at highest temperatures
        tmp_inp_array[:, 0] = tmp_n_inp[:, 0]
        for ii in range(1, self.ds["T"].size):
            tmp_inp_array[:, ii] = tmp_n_inp[:, ii] - tmp_inp_array[:, :ii].sum(axis=1)
        if self.singular_scale != 1.:
            tmp_inp_array *= self.singular_scale

        self.ds["T"].attrs["units"] = "$K$"  # set coordinate attributes.

        self.ds["inp"] = xr.DataArray(np.zeros((self.ds["height"].size, self.ds["time"].size, self.ds["T"].size)),
                                      dims=("height", "time", "T"))
        self.ds["inp"].loc[{"time": 0}] = np.flip(tmp_inp_array, axis=1)
        self.ds["inp"].attrs["units"] = "$L^{-1}$"
        self.ds["inp"].attrs["long_name"] = "INP concentration per temperature bin"

    def _init_inp_Jhet_ABIFM_arrays(self, ci_model):
        """
        initialize the INP and Jhet arrays for ABIFM (height x time x diameter) assuming that dn_dlogD has been
        calculated and that the ci_model object was already generated (with delta_aw, etc.).

        Parameters
        ---------
        ci_model: ci_model class object
            Cloud-INP model object including all model initialization and prognosed field datasets.
        """
        self.ds["Jhet"] = 10.**(self.Jhet.c + self.Jhet.m * ci_model.ds["delta_aw"])  # calc Jhet
        self.ds["Jhet"].attrs["units"] = "$cm^{-2} s{-1}$"
        self.ds["Jhet"].attrs["long_name"] = "Heterogeneous ice nucleation rate coefficient"
        self.ds["inp"] = xr.DataArray(np.zeros((self.ds["height"].size, self.ds["time"].size,
                                                self.ds["diam"].size)),
                                      dims=("height", "time", "diam"))
        self.ds["inp"].loc[{"time": 0}] = np.tile(self.dn_dlogD, (self.ds["height"].size, 1))
        self.ds["inp"].attrs["units"] = "$L^{-1}$"
        self.ds["inp"].attrs["long_name"] = "INP concentration per diameter bin"

    def _weight_inp_prof(self, use_ABIFM=True):
        """
        apply weights on initial INP profile (weighting on n_init_max). If using singular then returning the
        weights profile
        Parameters
        ---------
        use_ABIFM: bool
            True - use ABIFM, False - use singular.

        Returns
        -------
        weight_prof_interp: np.ndarray (--singular--)
            weight profile with height coordinates
        """
        if not np.all([x in self.n_init_weight_prof.keys() for x in ["height", "weight"]]):
            raise KeyError('Weighting the INP profiles requires the keys "height" and "weight"')
        if not np.logical_and(len(self.n_init_weight_prof["height"]) > 1,
                              len(self.n_init_weight_prof["height"]) == len(self.n_init_weight_prof["weight"])):
            raise ValueError("weights and heights must have the same length > 1")
        if np.any(self.n_init_weight_prof["weight"] < 0.):
            raise ValueError("weight values must by > 0 (at least one negative value was entered)")
        if np.any(self.n_init_weight_prof["weight"] > 1.):
            print("At least one specified weight > 1 (max value = %.1f); normalizing weight profile such that \
                  max weight == 1" % self.n_init_weight_prof["weight"].max())
            self.n_init_weight_prof["weight"] = self.n_init_weight_prof["weight"] / \
                self.n_init_weight_prof["weight"].max()
        weight_prof_interp = np.interp(self.ds["height"], self.n_init_weight_prof["height"],
                                       self.n_init_weight_prof["weight"])
        if use_ABIFM:
            self.ds["inp"][{"time": 0}] = np.tile(np.expand_dims(weight_prof_interp, axis=1),
                                                  (1, self.ds["diam"].size)) * self.ds["inp"][{"time": 0}]
        else:  # Relevant for singular when considering particle diameters (e.g., D2010, D2015).
            return weight_prof_interp


class mono_INP(INP_pop):
    """
    Uniform (fixed) INP diameter.
    """
    def __init__(self, use_ABIFM, n_init_max, nucleus_type=None, name=None,
                 diam_cutoff=0., T_array=None, singular_fun=None, singular_scale=None,
                 psd={}, n_init_weight_prof=None, ci_model=None):
        """
        Parameters as in the 'INP_pop' class (fixed diameter can be specified in the 'psd' dict under the 'diam'
        key or in the diam).
        """
        psd.update({"type": "mono"})  # require type key consistency
        if "diam" not in psd.keys():
            raise KeyError('mono-dispersed PSD processing requires the "diam" fields')
        diam = psd["diam"]
        dn_dlogD = np.array(n_init_max)
        super().__init__(use_ABIFM=use_ABIFM, n_init_max=n_init_max, nucleus_type=nucleus_type, diam=diam,
                         dn_dlogD=dn_dlogD, name=name, diam_cutoff=diam_cutoff, T_array=T_array,
                         singular_fun=singular_fun, singular_scale=singular_scale, psd=psd,
                         n_init_weight_prof=n_init_weight_prof, ci_model=ci_model)


class logn_INP(INP_pop):
    """
    Log-normal INP PSD.
    """
    def __init__(self, use_ABIFM, n_init_max, nucleus_type=None, name=None,
                 diam_cutoff=0., T_array=None, singular_fun=None, singular_scale=None,
                 psd={}, n_init_weight_prof=None, ci_model=None):
        """
        Parameters as in the 'INP_pop' class

        psd parameters
        --------------
        diam_mean: float
            geometric mean diameter [um]
        geom_sd: float
            geometric standard deviation
        n_bins: int
            number of bins in psd array
        diam_min: float
            minimum diameter [um]
        m_ratio: float
            bin-tp-bin mass ratio (smaller numbers give more finely resolved grid).
            Effectively, the diameter ratio between consecutive bins is m_ratio**(1/3).
        """
        psd.update({"type": "logn"})  # require type key consistency
        if not np.all([x in psd.keys() for x in ["diam_mean", "geom_sd", "n_bins", "diam_min", "m_ratio"]]):
            raise KeyError('log-normal PSD processing requires the fields' +
                           '"diam_mean", "geom_sd", "n_bins", "diam_min", "m_ratio"')
        diam, dn_dlogD = self._calc_logn_diam_dn_dlogd(psd, n_init_max)
        super().__init__(use_ABIFM=use_ABIFM, n_init_max=n_init_max, nucleus_type=nucleus_type, diam=diam,
                         dn_dlogD=dn_dlogD, name=name, diam_cutoff=diam_cutoff, T_array=T_array,
                         singular_fun=singular_fun, singular_scale=singular_scale, psd=psd,
                         n_init_weight_prof=n_init_weight_prof, ci_model=ci_model)

    def _calc_logn_diam_dn_dlogd(self, psd, n_init_max):
        """
        Assign particle diameter array and calculate dn_dlogD for log-normal distribution.

        Parameters
        ---------
        psd: dict
            Log-normal PSD parameters.
        n_init_max: float
            total initial INP concentration [L-1].

        Returns
        -------
        diam: np.ndarray
            Particle diameter array.
        dn_dlogD: np.ndarray
            Particle number concentration per diameter bin.
        """
        diam = np.ones(psd["n_bins"]) * psd["diam_min"]
        diam = diam * (psd["m_ratio"] ** (1. / 3.)) ** (np.cumsum(np.ones(psd["n_bins"])) - 1)
        denom = np.sqrt(2 * np.pi) * np.log(psd["geom_sd"])
        argexp = np.log(diam / psd["diam_mean"]) / np.log(psd["geom_sd"])
        dn_dlogD = (1 / denom) * np.exp(-0.5 * argexp**2)
        dn_dlogD = dn_dlogD / dn_dlogD.sum() * n_init_max
        return diam, dn_dlogD


class multi_logn_INP(logn_INP):
    """
    Multiple log-normal INP PSD.
    """
    def __init__(self, use_ABIFM, n_init_max, nucleus_type=None, name=None,
                 diam_cutoff=0., T_array=None, singular_fun=None, singular_scale=None,
                 psd={}, n_init_weight_prof=None, ci_model=None):
        """
        Parameters as in the 'INP_pop' class. Note that n_init_max should be a list or np.ndarray
        of values for each mode with the same length as diam_mean and geom_sd. Array bins are specified
        using scalars.

        psd parameters
        --------------
        diam_mean: list or np.ndarray of float
            geometric mean diameter [um] for each model
        geom_sd: list or np.ndarray of float
            geometric standard deviation for each mode
        n_bins: int
            number of bins in psd array
        diam_min: float
            minimum diameter [um]
        m_ratio: float
            bin-tp-bin mass ratio (smaller numbers give more finely resolved grid).
            Effectively, the diameter ratio between consecutive bins is m_ratio**(1/3).
        """
        psd.update({"type": "multi_logn"})  # require type key consistency
        if not np.all([x in psd.keys() for x in ["diam_mean", "geom_sd", "n_bins", "diam_min", "m_ratio"]]):
            raise KeyError('log-normal PSD processing requires the fields' +
                           '"diam_mean", "geom_sd", "n_bins", "diam_min", "m_ratio"')
        if not len(np.unique((len(n_init_max), len(psd["diam_mean"]), len(psd["geom_sd"])))) == 1:
            raise IndexError("'n_init_max', 'diam_mean', and 'geom_sd' must have the same length (one " +
                             "value for each mode)")
        for ii in range(len(n_init_max)):
            psd_tmp = psd.copy()
            psd_tmp["diam_mean"] = psd_tmp["diam_mean"][ii]
            psd_tmp["geom_sd"] = psd_tmp["geom_sd"][ii]
            diam_tmp, dn_dlogD_tmp = super()._calc_logn_diam_dn_dlogd(psd_tmp, n_init_max[ii])
            if ii == 0:
                diam, dn_dlogD = diam_tmp, dn_dlogD_tmp
            else:
                dn_dlogD += dn_dlogD_tmp
        super(logn_INP, self).__init__(use_ABIFM=use_ABIFM, n_init_max=np.sum(n_init_max),
                                       nucleus_type=nucleus_type, diam=diam, dn_dlogD=dn_dlogD, name=name,
                                       diam_cutoff=diam_cutoff, T_array=T_array, singular_fun=singular_fun,
                                       singular_scale=singular_scale, psd=psd,
                                       n_init_weight_prof=n_init_weight_prof, ci_model=ci_model)


class custom_INP(INP_pop):
    """
    custom INP PSD ('dn_dlogD' and 'diam' with optional normalization to n_init_max).
    """
    def __init__(self, use_ABIFM, n_init_max=None, nucleus_type=None, name=None,
                 diam_cutoff=0., T_array=None, singular_fun=None, singular_scale=None,
                 psd={}, n_init_weight_prof=None, ci_model=None):
        """
        Parameters as in the 'INP_pop' class

        psd parameters
        --------------
        norm_to_n_init_max: bool
            If True then dn_dlogD is normalized such that sum(dn_dlogD) = n_init_max. In that case, n_init_max
            must be specified.
        """
        psd.update({"type": "custom"})  # require type key consistency
        if not np.all([x in psd.keys() for x in ["diam", "dn_dlogD"]]):
            raise KeyError('custom PSD processing requires the fields "diam", "dn_dlogD"')
        diam = psd["diam"]
        dn_dlogD = psd["dn_dlogD"]
        if len(dn_dlogD) != len(diam):
            raise ValueError("The 'diam' and 'dn_dlogD' arrays must have the same size!")
        if "norm_to_n_init_max" in psd.keys():
            if psd["norm_to_n_init_max"]:
                dn_dlogD = dn_dlogD / np.sum(dn_dlogD) * n_init_max
        super().__init__(use_ABIFM=use_ABIFM, n_init_max=n_init_max, nucleus_type=nucleus_type, diam=diam,
                         dn_dlogD=dn_dlogD, name=name, diam_cutoff=diam_cutoff, T_array=T_array,
                         singular_fun=singular_fun, singular_scale=singular_scale, psd=psd,
                         n_init_weight_prof=n_init_weight_prof, ci_model=ci_model)
