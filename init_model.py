"""
This module is used to initialize the model, and allocate fields and arrays to a 'ci_model' class.
"""
import xarray as xr
import numpy as np
from time import time
import LES
import INP
import plotting
from run_model import run_model as Run


class ci_model():
    """
    Cloud-INP 1D model class containing:
    1. All initialization model parameters
    2. LES output dataset used to initialize and inform the model (ci_model.les).
    3. Model output  output fields (ci_model.ds).
    """
    def __init__(self, final_t=21600, delta_t=10, use_ABIFM=True, les_name="DHARMA", t_averaged_les=True,
                 custom_vert_grid=None, w_e_ent=0.1e-3, entrain_from_cth=True, tau_mix=1800.,
                 mixing_bounds=None, v_f_ice=0.3, in_cld_q_thresh=1e-3,
                 inp_info=None, les_out_path=None, les_out_filename=None, t_harvest=10800,
                 fields_to_retain=None, height_ind_2crop="ql_pbl", cbh_det_method="ql_cbh",
                 do_entrain=True, do_mix_aer=True, do_mix_ice=True, do_sedim=True, run_model=True):
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
        w_e_ent: dict or float
            cloud-top entrainment rate [m/s].
            if a float then using its value throughout the simulation time.
            if a dict, must have the keys "time" [s] and "value". Each key contains a list or np.ndarray of
            length s (s > 1) determining time and entrainment rate time series.
            Time values are interpolated between the specified times, and the edge values are used for
            extrapolation.
        entrain_from_cth: bool
            If True, then entrain from cloud top definition consistent with the 'cbh_det_method' input parameter.
            If False, then entraining from domain top.
        tau_mix: dict or float
            boundary-layer mixing time scale [s].
            if a float then using its value throughout the simulation time.
            if a dict, then treated as in the case of a dict for w_e_ent.
        mixing_bounds: two-element tuple or list, or None
            Determining the mixing layer (especially relevant when using time-varying LES input).
            The first element provides a fixed lowest range of mixing (float), a time varying range (dict as
            in w_e_ent), or the method with which to determine mixing base (str). The second element is
            similar, but for the determination of the mixing layer top.
            If None, using the full domain.
            NOTE: currently, the only accepted pre-specified mixing determination method is "ql_cbh"
            (q_liq-based cloud base or top height detection method, allowing limit mixing to the cloud).
        v_f_ice: xr DataArray, dict, or float
            number-weighted ice crystal fall velocity [m/s].
            if a float then using its value throughout the simulation time.
            if a dict, then treated as in the case of a dict for w_e_ent.
            if an xr DataArray, must contain the "height" [m] and "time" [s] coordinates. Values outside the
            coordinate range are extrapolated using the nearest edge values.
        in_cld_q_thresh: float
            Mixing ratio threshold [g/kg] for determination of in-cloud environment; also assigned to the
            'q_liq_pbl_cut' attribute value.
        inp_info: list of dict
            Used to initialize the INP arrays. Each element of the list describes a single population of an INP
            type providing its composition, concentration, and PSD, e.g., can use a single log-normal population
            of Illite, or two Illite PSDs with different mode diameter and geometric SD combined with a Kaolinite
            population.
            Each dictionary (i.e., an 'inp_attrs' list element) must contain the keys:

                1. n_init_max: [float] total concentration [L-1].

                2. psd: [dict] choose a 'type' key between several options (parentheses denote required dict key
                names):
                    - "mono": fixed-size population, i.e., a single particle diameter should be provided
                      (diam [um]).
                    - "logn": log--normal: provide geometric mean diameter (diam_mean [um]), geometric SD
                      (geom_sd), number of PSD bins (n_bins), minimum diameter (diam_min [um]) and
                      bin-to-bin mass ratio (m_ratio). Note that the effective bin-to-bin diameter ratio
                      equals m_ratio**(1/3).
                    - "custom": custom size distribution with maunally specified bin values and PSD shape.
                      Provide the PSD diameter array (diam) and the number concentration per bin
                      (dn_dlogD). Optional input key includes normalization to n_init (norm_to_n_init_max)
                      that normalizes dn_dlogD such that such sum(dn_dlogD) = n_init_max.
                    - "default": (parameters not required) using a log-normal PSD with mean diameter
                      of 1 um, geometric SD of 2.5, 35 PSD bins with minimum diameter of 0.01 um and mass
                      ratio of 2, resulting in max diameter of ~26 um.
            optional keys:
                1. name: [str] population name (or tag). A default string using nucleus type is used if not
                provided.

                2. nucleus_type: [str; --ABIFM--]  name of substance (e.g., Al2O3) - to initialize Jhet (must be
                specified for ABIFM).

                3. diam_cutoff: [float or tuple; --singular--] minimum particle diameter to consider.
                Using a value of 0.5 as in D2010 if not specified. Use a 2-element tuple to specify a range of
                diameters to consider.

                4. T_array: [list or np.ndarray; --singular--] discrete temperature array. If not specified, using
                temperatures between -40 and 0 with delta_T = 0.1 C.

                5. singular_fun: [lambda func. or str; --singular--] INP parametrization (typically as a function
                of T).
                str: use "D2010" to use eq. 1 in DeMott et al., 2010, "D2015" to use eq. 2 in DeMott et al.,
                2015, or "D2010fit" to use the temperature dependence fit from fig. 2 in DeMott et al., 2010.
                The D2015 has default values of the five coeff. from eq. 2 (cf - calibration correction factor,
                alpha, beta, gamma, delta); these might be coded as optional input for INP the class in
                the future.
                Note that "D2010fit" does not consider aerosol PSDs.
                Use "D2010" (default) if None.

                6. singular_scale: [float; --singular--] Scale factor for 'singular_fun' (1 by default).

                7. n_init_weight_prof: [dict] a dict with keys "height" and "weight". Each key contains
                a list or np.ndarray of length s (s > 1) determining PSD heights [m] and weighting profiles.
                Weights are applied on n_init such that n_init(z) = n_init_max * weighting_factor(z), i.e., a
                weighted_inp_prof filled with ones means that n_init(z) = n_init_max.
                if weights > 1 are specified, the profile is normalized to max value == 1. heights are interpolated
                between the specified heights, and the edge values are used for extrapolation (can be used to set
                different INP source layers at model initialization, and combined with turbulence weighting,
                allows the emulation of cloud-driven mixing.
        do_entrain: bool
            determines whether aerosols (INP) entrainment will be performed.
        do_mix_aer: bool
            determines whether mixing of aerosols (INP) will be performed.
        do_mix_ice: bool
            determines whether mixing of ice will be performed.
        do_sedim: bool
            determines whether ice sedimentation will be performed.
        run_model: bool
            True - run model once initialization is done.

        Other Parameters
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
        # count processing time
        Now = time()

        # Set some simulation attributes.
        self.vars_harvested_from_les = ["RH", "ql", "T", "Ni", "prec"]  # processed variables used by the model.
        self.final_t = final_t
        self.use_ABIFM = use_ABIFM
        self.in_cld_q_thresh = in_cld_q_thresh  # g/kg

        # Load LES output
        if les_name == "DHARMA":
            les = LES.DHARMA(les_out_path=les_out_path, les_out_filename=les_out_filename, t_harvest=t_harvest,
                             fields_to_retain=fields_to_retain, height_ind_2crop=height_ind_2crop,
                             cbh_det_method=cbh_det_method, q_liq_pbl_cut=in_cld_q_thresh)
        else:
            raise NameError("Can't process LES model output from '%s'" % les_name)
        self.LES_attributes = {"LES_name": les_name,
                               "les_out_path": les.les_out_path,
                               "les_out_filename": les.les_out_filename,
                               "t_averaged_les": t_averaged_les,
                               "t_harvest": t_harvest,
                               "fields_to_retain": fields_to_retain,
                               "height_ind_2crop": height_ind_2crop,
                               "cbh_det_method": cbh_det_method}

        # time-averaged LES variable profile option
        if t_averaged_les:
            les_units = {}
            for key in self.vars_harvested_from_les:
                les_units.update({key: les.ds[key].attrs["units"]})
            Mean_time = les.ds["time"].mean()
            les.ds = les.ds.mean(dim="time")
            les.ds = les.ds.assign_coords({"time": Mean_time})
            les.ds = les.ds.expand_dims("time").transpose(*("height", "time"))
            for key in self.vars_harvested_from_les:  # restore attributes lost during averaging.
                les.ds[key].attrs["units"] = les_units[key]

            # Redetermine cloud bounds with the time-averaged profile for model consistency (entrainment, etc.).
            tmp_ds = xr.Dataset()  # first, use a temporary xr.Dataset to retain t-averaged precip rates.
            tmp_ds["P_Ni"], tmp_ds["Pcb_per_Ni"] = les.ds["P_Ni"].copy(), les.ds["Pcb_per_Ni"].copy()
            les._find_and_calc_cb_precip(self.LES_attributes["cbh_det_method"])
            tmp_fields = [x for x in les.ds.keys()]
            les.ds["P_Ni"].values, les.ds["Pcb_per_Ni"].values = tmp_ds["P_Ni"].values, tmp_ds["Pcb_per_Ni"].values

            # crop updated dataset (temporarily change les object attributes to invoke internal method)
            tmp_attrs = {"ql": les.q_liq_field, "height_dim": les.height_dim}
            les.q_liq_field["name"], les.q_liq_field["scaling"], les.height_dim = "ql", 1, "height"
            les._crop_fields(tmp_fields, height_ind_2crop)
            les.q_liq_field["name"], les.q_liq_field["scaling"], les.height_dim = \
                tmp_attrs["ql"]["name"], tmp_attrs["ql"]["scaling"], tmp_attrs["height_dim"]

        # Retain only the LES xr.Dataset for accessibility
        self.les = les.ds

        # Make sure ice does not sediment more than 1 vertical cell per time step. In that case change delta_t
        if isinstance(v_f_ice, dict):
            max_sediment_vel = np.max(v_f_ice["value"])
        else:
            max_sediment_vel = np.max(v_f_ice)
        max_sediment_dist = max_sediment_vel * delta_t  # maximum ice sedimentation distance per time step
        if custom_vert_grid is not None:
            height = custom_vert_grid.astype(np.float32)
            height = height[np.logical_and(height <= self.les["height"].max().values,
                                           height >= self.les["height"].min().values)]
            if len(height) < len(custom_vert_grid):
                print("Some heights were omitted because they are outside the processed LES dataset grid")
        else:
            height = self.les["height"].values
        if max_sediment_dist > np.min(np.diff(height)):
            delta_t = np.floor(np.min(np.diff(height)) / max_sediment_vel)
            print("∆t was modified to the largest integer preventing ice sedimentation of more than 1 " +
                  "grid cell (%d s)" % delta_t)
        self.delta_t = delta_t
        self.mod_nt = int(final_t / delta_t) + 1  # number of time steps
        self.mod_nz = len(height)  # number of vertical layers

        # allocate xarray DataSet for model atmospheric state and prognosed variable fields
        self.ds = xr.Dataset()
        self.ds = self.ds.assign_coords({"height": height})
        self.ds = self.ds.assign_coords({"time": np.arange(self.mod_nt) * self.delta_t})
        delta_z = np.diff(self.ds["height"])
        self.ds["delta_z"] = xr.DataArray(np.concatenate((delta_z, np.array([delta_z[-1]]))),
                                          dims=("height"), attrs={"units": "$m$"})
        extrap_locs_tail = self.ds["time"] >= self.les["time"].max()
        extrap_locs_head = self.ds["time"] <= self.les["time"].min()
        x, y = np.meshgrid(self.les["height"], self.les["time"])
        for key in self.vars_harvested_from_les:

            # Linear interp (two 1D interpolations - fastest) if LES temporal evolution is to be considered.
            if self.les["time"].size > 1:
                self._set_1D_or_2D_var_from_input(self.les[key], key)
            else:
                # Use LES bounds (min & max) outside the available range (redundant step - could be useful later).
                key_array_tmp = np.zeros((self.mod_nz, self.mod_nt))
                if extrap_locs_head.sum() > 0:
                    key_array_tmp[:, extrap_locs_head.values] = np.tile(np.expand_dims(
                        np.interp(self.ds["height"], self.les["height"],
                                  self.les[key].sel({"time": self.les["time"].min()})),
                        axis=1), (1, np.sum(extrap_locs_head.values)))
                if extrap_locs_tail.sum() > 0:
                    key_array_tmp[:, extrap_locs_tail.values] = np.tile(np.expand_dims(
                        np.interp(self.ds["height"], self.les["height"],
                                  self.les[key].sel({"time": self.les["time"].max()})),
                        axis=1), (1, np.sum(extrap_locs_tail.values)))
                self.ds[key] = xr.DataArray(key_array_tmp, dims=("height", "time"))
            self.ds[key].attrs = self.les[key].attrs

        # init entrainment
        self.w_e_ent = w_e_ent
        self.entrain_from_cth = entrain_from_cth
        self._set_1D_or_2D_var_from_input(w_e_ent, "w_e_ent", "$m/s$", "Cloud-top entrainment rate")
        if entrain_from_cth:  # add cloud-top height for entrainment calculations during model run.
            if self.les["time"].size > 1:
                self._set_1D_or_2D_var_from_input({"time": self.les["time"].values,
                                                   "value": self.les["lowest_cth"].values},
                                                  "lowest_cth", "$m$", "Lowest cloud top height")
            else:
                self._set_1D_or_2D_var_from_input(self.les["lowest_cth"].item(),
                                                  "lowest_cth", "$m$", "Lowest cloud top height")

        # init vertical mixing and generate a mixing layer mask for the model
        self.tau_mix = tau_mix
        self._set_1D_or_2D_var_from_input(tau_mix, "tau_mix", "$s$", "Boundary-layer mixing time scale")
        if mixing_bounds is None:
            self.ds["mixing_mask"] = xr.DataArray(np.full((self.mod_nz, self.mod_nt),
                                                          True, dtype=bool), dims=("height", "time"))
        else:
            if isinstance(mixing_bounds[0], str):
                if mixing_bounds[0] == "ql_cbh":
                    self.ds["mixing_base"] = xr.DataArray(np.interp(
                        self.ds["time"], self.les["time"], self.les["lowest_cbh"]), dims=("time"))
                    self.ds["mixing_base"].attrs["units"] = "$m$"
            else:
                self._set_1D_or_2D_var_from_input(mixing_bounds[0], "mixing_base", "$m$", "Mixing layer base")
            if isinstance(mixing_bounds[1], str):
                if mixing_bounds[1] == "ql_cbh":
                    self.ds["mixing_top"] = xr.DataArray(np.interp(
                        self.ds["time"], self.les["time"], self.les["lowest_cth"]), dims=("time"))
                    self.ds["mixing_top"].attrs["units"] = "$m$"
            else:
                self._set_1D_or_2D_var_from_input(mixing_bounds[1], "mixing_top", "$m$", "Mixing layer top")
            mixing_mask = np.full((self.mod_nz, self.mod_nt), False, dtype=bool)
            for t in range(self.mod_nt):
                rel_ind = np.arange(
                    np.argmin(np.abs(self.ds["height"].values - self.ds["mixing_base"].values[t])),
                    np.argmin(np.abs(self.ds["height"].values - self.ds["mixing_top"].values[t])))
                mixing_mask[rel_ind, t] = True
            self.ds["mixing_mask"] = xr.DataArray(mixing_mask, dims=("height", "time"))
        self.ds["mixing_mask"].attrs["long_name"] = "Mixing-layer mask (True --> mixed)"

        # init number weighted ice fall velocity
        self.v_f_ice = v_f_ice
        self._set_1D_or_2D_var_from_input(v_f_ice, "v_f_ice", "$m/s$", "Number-weighted ice crystal fall velocity")

        # calculate delta_aw
        self._calc_delta_aw()

        # allocate for INP population Datasets
        self.inp = {}
        self.inp_info = inp_info  # save the INP info dict for reference.
        optional_keys = ["name", "nucleus_type", "diam_cutoff", "T_array",  # optional INP class input parameters.
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

        # allocate nucleated ice DataArrays
        self.ds["Ni_nuc"] = xr.DataArray(np.zeros((self.mod_nz,
                                         self.mod_nt)), dims=("height", "time"))
        self.ds["Ni_nuc"].attrs["units"] = "$L^{-1}$"
        self.ds["Ni_nuc"].attrs["long_name"] = "Nucleated ice"
        self.ds["nuc_rate"] = xr.DataArray(np.zeros((self.mod_nz,
                                           self.mod_nt)), dims=("height", "time"))
        self.ds["nuc_rate"].attrs["units"] = "$L^{-1}\:s^{-1}$"
        self.ds["nuc_rate"].attrs["long_name"] = "Ice nucleation rate"

        print("Model initalization done! Total processing time = %f s" % (time() - Now))

        # Set coordinate attributes
        self.ds["height"].attrs["units"] = "$m$"
        self.ds["time"].attrs["units"] = "$s$"
        self.ds["time_h"] = self.ds["time"].copy() / 3600  # add coordinates for time in h.
        self.ds["time_h"].attrs["units"] = "$h$"

        # Add plotting routines
        self.plot = plotting

        # Run the model
        self.do_entrain = do_entrain
        self.do_mix_aer = do_mix_aer
        self.do_mix_ice = do_mix_ice
        self.do_sedim = do_sedim
        if run_model:
            Run(self)

    def _calc_delta_aw(self):
        """
        calculate the ∆aw field for ABIFM using:
        1. eq. 1 in Knopf and Alpert (2013, https://doi.org/10.1039/C3FD00035D) combined with:
        2. eq. 7 in Koop and Zobrist (2009, https://doi.org/10.1039/B914289D) for a_w(ice)
        Here we assume that our droplets are in equilibrium with the environment at its given RH, hence, RH = a_w.
        """
        self.ds["delta_aw"] = self.ds['RH'] - \
            (
            np.exp(9.550426 - 5723.265 / self.ds['T'] + 3.53068 * np.log(self.ds['T']) -
                   0.00728332 * self.ds['T']) /
            (np.exp(54.842763 - 6763.22 / self.ds['T'] -
             4.210 * np.log(self.ds['T']) + 0.000367 * self.ds['T'] +
             np.tanh(0.0415 * (self.ds['T'] - 218.8)) * (53.878 - 1331.22 / self.ds['T'] - 9.44523 *
                                                         np.log(self.ds['T']) + 0.014025 * self.ds['T'])))
        )
        self.ds['delta_aw'].attrs['units'] = ""

    def _set_1D_or_2D_var_from_input(self, var_in, var_name, units_str=None, long_name_str=None):
        """
        set a 1D xr.DataArray from a scalar or a dictionary containing "time" and "value" keys.
        If 'var_in' is a scalar then generating a uniform time series.
        Values are linearly interpolated onto the model temporal grid (values outside the provided
        range are extrapolated.
        The method can also operate on an xr.DataArray. In that case it interpolates the input
        variable (containing "time" and "height" coordinates) onto the ci_model object's grid
        and also extrapolates using edge values (two-1D linear interpolations are performed).

        Parameters
        ---------
        var_in: xr.DataArray, dict, or scalar.
            if xr.DataArray, must have "time" and "height" coordinates and dims.
            if dict then using the "time" and "value" keys of the variable.
        var_name: str
            Name of DataArray variable.
        units_str: str
            string for the units attribute.
        long_name_str: str
            string for the long_name attribute.
        """
        if isinstance(var_in, (float, int)):
            self.ds[var_name] = xr.DataArray(np.ones(self.mod_nt) * var_in, dims=("time"))
        elif isinstance(var_in, dict):  # 1D linear interpolation
            if not np.all([x in var_in.keys() for x in ["time", "value"]]):
                raise KeyError('variable time series requires the keys "time" and "value"')
            if not np.logical_and(len(var_in["time"]) > 1,
                                  len(var_in["time"]) == len(var_in["value"])):
                raise ValueError("times and values must have the same length > 1")
            self.ds[var_name] = xr.DataArray(np.interp(self.ds["time"],
                                             var_in["time"], var_in["value"]), dims=("time"))
        elif isinstance(var_in, xr.DataArray):  # 2D linear interpolation
            if not np.all([x in var_in.coords for x in ["time", "height"]]):
                raise KeyError('2D variable processing requires the "time" and "height" coordinates!')
            if not np.logical_and(len(var_in["time"]) > 1, len(var_in["height"]) > 1):
                raise ValueError("times and height coordinates must be longer than 1 for interpolation!")
            key_array_tmp = np.zeros((self.mod_nz, self.mod_nt))
            key_1st_interp = np.zeros((var_in["height"].size, self.mod_nt))
            for hh in range(var_in["height"].size):
                key_1st_interp[hh, :] = np.interp(self.ds["time"].values, var_in["time"].values,
                                                  var_in.isel({"height": hh}))
            for tt in range(self.mod_nt):
                key_array_tmp[:, tt] = np.interp(self.ds["height"].values, var_in["height"].values,
                                                 key_1st_interp[:, tt])
            self.ds[var_name] = xr.DataArray(key_array_tmp, dims=("height", "time"))
        else:
            raise TypeError("Input variable must be of type float, int, dict, or xr.DataArray!")
        if units_str is not None:
            self.ds[var_name].attrs["units"] = units_str
        if long_name_str is not None:
            self.ds[var_name].attrs["long_name"] = long_name_str

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
