"""
This file is used to initialize the model, and allocate fields and arrays to a 'ci_model' class.
"""
import xarray as xr
import numpy as np
from scipy.interpolate import interp2d
from time import time
import LES
import INP


class ci_model():
    """
    Cloud-INP 1D model class containing:
    1. All initialization model parameters
    2. LES output dataset used to initialize and inform the model (ci_model.les).
    3. Model output  output fields (ci_model.ds).
    """
    def __init__(self, final_t=3600*6, delta_t=10, use_ABIFM=True, les_name="DHARMA", t_averaged_les=True,
                 custom_vert_grid=None, inp_info=None,
                 les_out_path=None, les_out_filename=None, t_harvest=3600*3,
                 fields_to_retain=None, height_ind_2crop="ql_pbl", cbh_det_method="ql_cbh"):
        """
        Model namelists and unit conversion coefficient required for the 1D model.
        The LES class includes methods to processes model output and prepare the out fields for the 1D model.

        Parameters
        ----------
        final_t: float
            Total simulation time [s].
        delta_t: float
            time_step [s].
        use_ABIFM: bool
            True - use ABIFM, False - use singular.
        les_name: str
            Name of LES model to harvest data from.
        t_averaged_les: bool
            If True, use time-averaged LES profile of each variable to inform the 1D model.
            If False then the 1D model is informed by the LES output temporal evolution with extrapolation
            outside the LES output DataSet time range.
            Note: in the case of a single LES output time step requested ('t_harvest' is a scalar), this boolean
            has no effect.
        custom_vert_grid: list, np.ndarray, or None.
            custom vertical grid for the 1D model. If None, then using the processed (and cropped) LES output
            grid.
        inp_info: list of dict
            Used to initialize the INP arrays. Each element of the list describes a single population of an INP
            type providing its composition, concentration, and PSD, e.g., can use a single log-normal population
            of Illite, or two Illite PSDs with different mode diameter and geometric SD combined with a Kaolinite
            population.
            Each dictionary (i.e., an 'inp_attrs' list element) must contain the keys:
            1. n_init_max: [float] total concentration [L-1].
            2. psd:   [dict] choose a 'type' key between several options (parentheses denote required dict key
                             names):
                            - "mono": fixed-size population, i.e., a single particle diameter should be provided
                              (diam).
                            - "logn": log--normal: provide geometric mean diameter (diam_mean), geometric SD
                              (geom_sd), number of PSD bins (n_bins), minimum diameter (diam_min) and bin-to-bin
                              mass ratio (m_ratio). Note that the effective bin-to-bin diameter ratio equals
                              m_ratio**(1/3)
                            - "custom": custom size distribution with maunally specified bin values and PSD shape.
                              Provide the PSD diameter array (diam) and the number concentration per bin
                              (dn_dlogD). Optional input key includes normalization to n_init (norm_to_n_init_max)
                              that normalizes dn_dlogD such that such sum(dn_dlogD) = n_init_max.
                            - "default": (parameters not required) using a log-normal PSD with mean diameter
                              of 1 um, geometric SD of 2.5, 35 PSD bins with minimum diameter of 0.01 um and mass
                              ratio of 2, resulting in max diameter of ~26 um.
            optional keys:
            1. name: [str] population name (or tag). A default string using nucleus type is used if not provided.
            2. nucleus_type: [str; --ABIFM--]  name of substance (e.g., Al2O3) - to initialize Jhet (must be
               specified for ABIFM).
            3. diam_cutoff: [float; --singular--] minimum particle diameter to consider (--singular--). Using a
               value of 0 (all diameters are considered) if not specified.
            4. T_array: [list or np.ndarray; --singular--] discrete temperature array. If not specified, using
               temperatures between -40 and 0 with delta_T = 0.1 C.
            5. singular_fun: [lambda function; --singular--] INP parametrization (typically as a function of T).
                    Uses DeMott et al., 2010 if None.
            6. singular_scale: [float; --singular--] Scale factor for 'singular_fun' (1 by default).
            7. n_init_weight_prof: [dict; --ABIFM--] a dict with keys "height" and "weight". Each key contains a
               list or np.ndarray of length s (s > 1) determining PSD heights [m] and weighting profiles. Weights
               are applied on n_init such that n_init(z) = n_init_max * weighting_factor(z), i.e., a
               weighted_inp_prof filled with ones means that n_init(z) = n_init_max.
               All weighting values greater (smaller) than 1 (0) are set to 1 (0). heights are interpolated
               between the specified heights, and the edge values are used for extrapolation (can be used to set
               different INP source layers at model initialization, and combined with turbulence weighting,
               allows the emulation of cloud-driven mixing.

        LES-related parameters
        ----------------------
        les_out_path: str or None
            LES output path (can be relative to running directory). Use default if None.
        les_out_filename: str or None
            LES output filename. Use default file if None.
        t_harvest: scalar, 2-element tuple, list (or ndarray), or None
            If scalar then using the nearest time (assuming units of seconds) to initialize the model
            (single profile).
            If a tuple, cropping the range defined by the first two elements (increasing values) using a
            slice object.
            If a list, cropping the times specified in the list (can be used take LES output profiles every
            delta_t seconds.
            NOTE: default in the ci_model class (10800 s) is different than in the DHARMA init method (None).
        fields_to_retain: list or None
            Fieldnames to crop from the LES output (required to properly run the model).
            If None, then cropping the minimum number of required fields using DHARMA's namelist convention
            (Temperature [K], q_liq [kg/kg], RH [fraction], precipitation flux [mm/h], and ice number
            concentration [cm^-3]).
        height_ind_2crop: list, str, or None
            Indices of heights to crop from the model output (e.g., up to the PBL top).
            if str then different defitions for PBL:
                - if == "ql_pbl" then cropping all values within the PBL defined here based on the
                'q_liq_pbl_cut' attribute. If more than a single time step exist in the dataset, then cropping
                the highest index corresponding to the cutoff.
                - OTHER OPTIONS TO BE ADDED.
            If None then not cropping.
            NOTE: default in the ci_model class ("ql_pbl") is different than in the DHARMA init method (None).
        cbh_det_method: str
            Method to determine cloud base with:
                - if == "ql_cbh" then cbh is determined by a q_liq threshold set with the 'q_liq_cbh' attribute.
                - OTHER OPTIONS TO BE ADDED.
        """
        self.vars_harvested_from_les = ["RH", "ql", "T", "Ni", "prec"] # variables used by the model (after LES processing).
        self.final_t = final_t
        self.delta_t = delta_t
        self.use_ABIFM = use_ABIFM
        self.mod_nt  = int(final_t/delta_t)+1  # number of time steps

        # Load LES output
        if les_name == "DHARMA":
            les = LES.DHARMA(les_out_path=les_out_path, les_out_filename=les_out_filename, t_harvest=t_harvest,
                             fields_to_retain=fields_to_retain, height_ind_2crop=height_ind_2crop,
                             cbh_det_method=cbh_det_method)
        else:
            raise NameError("Can't process LES model output from '%s'" % les_name)
        self.LES_attributes = {"LES_name": les_name,
                               "les_out_path": les.les_out_path,
                               "les_out_filename" : les.les_out_filename,
                               "t_harvest": t_harvest,
                               "fields_to_retain": fields_to_retain,
                               "height_ind_2crop": height_ind_2crop,
                               "cbh_det_method": cbh_det_method}
        self.les = les.ds

        # time-averaged LES variable profile option
        if t_averaged_les:
            les_units = {}
            for key in self.vars_harvested_from_les:
                les_units.update({key: self.les[key].attrs["units"]})
            Mean_time = self.les["time"].mean()
            self.les = self.les.mean(dim="time")
            self.les = self.les.assign_coords({"time": Mean_time})
            self.les = self.les.expand_dims("time").transpose(*("height", "time"))
            for key in self.vars_harvested_from_les:  # restore attributes lost during averaging.
                self.les[key].attrs["units"] = les_units[key]

        # allocate xarray DataSet for model atmospheric state variable fields
        self.ds = xr.Dataset()
        if custom_vert_grid is not None:
            height = custom_vert_grid
            height = height[np.logical_and(height <= self.les["height"].max().values,
                                             height >= self.les["height"].min().values)]
            if len(height) < len(custom_vert_grid):
                print("Some heights were omitted because they are outside the processed LES dataset grid")
        else:
            height = self.les["height"].values
        self.ds = self.ds.assign_coords({"height": height})
        self.ds = self.ds.assign_coords({"time": np.arange(self.mod_nt) * self.delta_t})
        vars_dim = (len(self.ds["height"].values), len(self.ds["time"]))
        extrap_locs_tail = self.ds["time"] >= self.les["time"].max()
        extrap_locs_head = self.ds["time"] <= self.les["time"].min()
        x, y = np.meshgrid(self.les["height"], self.les["time"])
        for key in self.vars_harvested_from_les:
            #self.ds[key] = xr.DataArray(np.zeros(vars_dim), dims=("height", "time"))
            key_array_tmp = np.zeros(vars_dim)

            # Linear interp (two 1D interpolations - fastest) if LES temporal evolution is to be considered.
            if self.les["time"].size > 1:
                print("Now performing 2D linear interpolation of %s - this may take some time" % key)
                Now = time()
                key_1st_interp = np.zeros((self.les["height"].size, self.ds["time"].size))
                for hh in range(self.les["height"].size):
                    key_1st_interp[hh, :] = np.interp(self.ds["time"].values, self.les["time"].values,
                        self.les[key].isel({"height": hh}))
                for tt in range(self.ds["time"].size):
                    key_array_tmp[:, tt] = np.interp(self.ds["height"].values, self.les["height"].values,
                        key_1st_interp[:, tt])
                print("Done! Interpolation time = %f s" % (time() - Now))
            else:
            # Use LES bounds (min and max) outside the available range (redundant step but could be useful later).
                if extrap_locs_head.sum() > 0:
                    key_array_tmp[:, extrap_locs_head.values] = \
                                np.tile(np.expand_dims(self.les[key].sel({"time": self.les["time"].min()}).values,
                                axis=1), (1, np.sum(extrap_locs_head.values)))
                if extrap_locs_tail.sum() > 0:
                    key_array_tmp[:, extrap_locs_tail.values] = \
                                np.tile(np.expand_dims(self.les[key].sel({"time": self.les["time"].max()}).values,
                                axis=1), (1, np.sum(extrap_locs_tail.values)))

            self.ds[key] = xr.DataArray(key_array_tmp, dims=("height", "time"))
            self.ds[key].attrs = self.les[key].attrs
        self.ds["height"].attrs["units"] = "m"
        self.ds["time"].attrs["units"] = "s"

        # calculate delta_aw
        self._calc_delta_aw()
        
        # allocate for INP population Datasets
        self.inp = {}
        optional_keys = ["name", "nucleus_type", "diam_cutoff", "T_array",  # list of optional INP class input parameters.
                         "n_init_weight_prof", "singular_fun", "singular_scale"]
        for ii in range(len(inp_info)):
            param_dict = {"use_ABIFM": use_ABIFM}  # tmp dict for INP attributes to send INP class call.
            if np.all([x in inp_info[ii].keys() for x in ["n_init_max", "psd"]]):
                param_dict["n_init_max"] = inp_info[ii]["n_init_max"]
                param_dict["psd"] = inp_info[ii]["psd"]
            else:
                raise KeyError('INP information requires the keys "n_init_max", "psd"')
            if not inp_info[ii]["psd"]["type"] in ["mono", "logn", "custom", "default"]:
                raise ValueError('PSD type must be one of: "mono", "logn", "custom", "default"')
            for key in optional_keys:
                param_dict[key] = inp_info[ii][key] if key in inp_info[ii].keys() else None

            # set INP population arrays
            tmp_inp_pop = self._set_inp_obj(param_dict)
            self.inp[tmp_inp_pop.name] = tmp_inp_pop

    def _calc_delta_aw(self):
        """
        calculate the ∆aw field for ABIFM using:
        1. eq. 1 in Knopf and Alpert (2013, https://doi.org/10.1039/C3FD00035D) combined with:
        2. eq. 7 in Koop and Zobrist (2009, https://doi.org/10.1039/B914289D) for a_w(ice)
        Here we assume that our droplets are in equilibrium with the environment at its given RH, hence, RH = a_w.
        """
        self.ds["delta_aw"] = self.ds['RH'] - \
                                 (
                                 np.exp(9.550426-5723.265 / self.ds['T'] + 3.53068 * np.log(self.ds['T']) -
                                 0.00728332 * self.ds['T']) / \
                                 (np.exp(54.842763 - 6763.22 / self.ds['T'] -
                                 4.210 * np.log(self.ds['T']) + 0.000367 * self.ds['T'] + np.tanh(0.0415 *
                                 (self.ds['T'] - 218.8)) * (53.878 - 1331.22 / self.ds['T'] - 9.44523 *
                                 np.log(self.ds['T']) + 0.014025 * self.ds['T'])))
                                 )
        self.ds['delta_aw'].attrs['units'] = ""

    def _set_inp_obj(self, param_dict):
        """
        Invoke an INP class call and use the input parameters provided. Using a full dictionary key call to
        maintain consistency even if some INP class input variable order will be changed in future updates.

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
        if param_dict["psd"]["type"] == "mono":
            tmp_inp_pop = INP.mono_INP(use_ABIFM=param_dict["use_ABIFM"], n_init_max=param_dict["n_init_max"],
                                       psd=param_dict["psd"], nucleus_type=param_dict["nucleus_type"],
                                       name=param_dict["name"], diam_cutoff=param_dict["diam_cutoff"],
                                       T_array=param_dict["T_array"], singular_fun=param_dict["singular_fun"],
                                       singular_scale=param_dict["singular_scale"],
                                       n_init_weight_prof=param_dict["n_init_weight_prof"], ci_model=self)
        elif param_dict["psd"]["type"] == "logn":
           tmp_inp_pop = INP.logn_INP(use_ABIFM=param_dict["use_ABIFM"], n_init_max=param_dict["n_init_max"],
                                       psd=param_dict["psd"], nucleus_type=param_dict["nucleus_type"],
                                       name=param_dict["name"], diam_cutoff=param_dict["diam_cutoff"],
                                       T_array=param_dict["T_array"], singular_fun=param_dict["singular_fun"],
                                       singular_scale=param_dict["singular_scale"],
                                       n_init_weight_prof=param_dict["n_init_weight_prof"], ci_model=self)
        elif param_dict["psd"]["type"] == "custom":
            tmp_inp_pop = INP.custom_INP(use_ABIFM=param_dict["use_ABIFM"], n_init_max=param_dict["n_init_max"],
                                       psd=param_dict["psd"], nucleus_type=param_dict["nucleus_type"],
                                       name=param_dict["name"], diam_cutoff=param_dict["diam_cutoff"],
                                       T_array=param_dict["T_array"], singular_fun=param_dict["singular_fun"],
                                       singular_scale=param_dict["singular_scale"],
                                       n_init_weight_prof=param_dict["n_init_weight_prof"], ci_model=self)
        elif param_dict["psd"]["type"] == "default":
            param_dict["psd"].update({"diam_mean": 1, "geom_sd": 2.5, "n_bins": 35, "diam_min": 0.01,
                                      "m_ratio": 2.})  # default parameters.
            tmp_inp_pop = INP.logn_INP(use_ABIFM=param_dict["use_ABIFM"], n_init_max=param_dict["n_init_max"],
                                       psd=param_dict["psd"], nucleus_type=param_dict["nucleus_type"],
                                       name=param_dict["name"], diam_cutoff=param_dict["diam_cutoff"],
                                       T_array=param_dict["T_array"], singular_fun=param_dict["singular_fun"],
                                       singular_scale=param_dict["singular_scale"],
                                       n_init_weight_prof=param_dict["n_init_weight_prof"], ci_model=self)

        return tmp_inp_pop
