"""
LES class and the inherited DHARMA class

"""
import xarray as xr
import numpy as np


class LES():
    """
    Model namelists and unit conversion coefficient required for the 1D model.
    The LES class includes methods to processes model output and prepare the out fields for the 1D model.

    Attributes
    ----------
    Ni_field: dict
       Ice number concentration fieldname and scaling factor required for L^-1 units
    pflux_field: dict
        Precipitation flux fieldname and scaling factor for mm/h.
    T_field: dict
        Temperature field name and addition factor in case of T reported in C (K is needed).
    q_liq_field: dict
        Liquid mixing ratio fieldname and scaling factor for g/kg.
    RH_field: dict
        RH fieldname and scaling factor for fraction units (not %).
    model_name: str
        name of model.
    time_dim: str
        name of the time dimension.
    height_dim: str
        name of the height dim.
    height_dim_2nd: str
        name of the height dim for grid cell edge coordinates.
    q_liq_pbl_cutoff: float
        value of q_liq cutoff (g/kg) required to define the PBL top z-index (where LWC becomes negligible) -
        to be used in '_crop_fields' if 'height_ind_2crop' == 'ql_pbl'.
    q_liq_cbh: float
        Threshold value (g/kg) for cloud base height defined using q_liq. 
    """
    def __init__(self):
        self.Ni_field = {"name": None, "scaling": None}  # scale to L^-1
        self.pflux_field = {"name": None, "scaling": None}  # scale to mm/h
        self.T_field = {"name": None, "addition": None}  # scale to K
        self.q_liq_field = {"name": None, "scaling": None}  # scale to g/kg
        self.RH_field = {"name": None, "scaling": None}  # scale to fraction
        self.model_name = ""
        self.time_dim = ""  # assuming in seconds
        self.height_dim = ""  # assuming in m
        self.height_dim_2nd = "" # assuming in m
        self.q_liq_pbl_cut = 1e-3  # g/kg (default value of 1e-3).
        self.q_liq_cbh = self.q_liq_pbl_cut  # Equal to the pbl cutoff value by default

    def _crop_time_range(self, t_harvest=None):
        """
        Crop model output time range.

        Parameters
        ----------
        t_harvest: scalar, 2-element tuple, list (or ndarray), or None
            If scalar then using the nearest time (assuming units of seconds) to initialize the model
            (single profile).
            If a tuple, cropping the range defined by the first two elements (increasing values) using a
            slice object.
            If a list, cropping the times specified in the list (can be used take LES output profiles every
            delta_t seconds.
        """
        if isinstance(t_harvest, (float,int)):
            self.ds = self.ds.sel({self.time_dim: [t_harvest]}, method='nearest') 
        elif isinstance(t_harvest, tuple):  # assuming a 2-element tuple.
            assert len(t_harvest) == 2, "t_harvest (time range) tuple length should be 2"
            self.ds = self.ds.sel({self.time_dim: slice(*t_harvest)})
        elif isinstance(t_harvest, (list, np.ndarray)):
            self.ds = self.ds.sel({self.time_dim: t_harvest}, method='nearest')

    def _crop_fields(self, fields_to_retain=None, height_ind_2crop=None):
        """
        Crop the required fields (and other requested fields), with the option of cropping the
        height dim using specified indices.

        Parameters
        ----------
        fields_to_retain: list or None
            Fieldnames to crop from the LES output (required to properly run the model).
            If None, then cropping the minimum number of required fields using the model's namelist convention
            (Temperature, q_liq, RH, precipitation flux, and ice number concentration).
        height_ind_2crop: list, str, or None
            Indices of heights to crop from the model output (e.g., up to the PBL top).
            if str then different defitions for PBL:
                - if == "ql_pbl" then cropping all values within the PBL defined here based on the
                'q_liq_pbl_cut' attribute. If more than a single time step exist in the dataset, then cropping
                the highest index corresponding to the cutoff.
                - OTHER OPTIONS TO BE ADDED.
            If None then not cropping.
        """
        if fields_to_retain is None:
            fields_to_retain = [self.Ni_field["name"], self.pflux_field["name"], self.T_field["name"],
                                self.q_liq_field["name"], self.RH_field["name"]]

        # crop variables
        self.ds = self.ds[fields_to_retain]  # retain variables needed.

        # crop heights
        if height_ind_2crop is not None:
            if isinstance(height_ind_2crop, str):
                if height_ind_2crop == "ql_pbl":  # 1D model domain extends only to PBL top
                    rel_inds = np.arange(np.max(np.where(
                                         self.ds[self.q_liq_field["name"]].values >= self.q_liq_pbl_cut /
                                         self.q_liq_field["scaling"])[0]) + 1)
                else:
                    print("Unknown croppoing method string - skipping xr dataset (LES domain) cropping.")
            elif isinstance(height_ind_2crop, (list, np.ndarray)):
                rel_inds = height_ind_2crop
            self.ds = self.ds[{self.height_dim: rel_inds}]

    def _prepare_les_dataset_for_1d_model(self, cbh_det_method="ql_cbh"):
        """
        scale (unit conversion), rename the required fields (prepare the dataset for informing the 1D model
        allowing retaining only the les xr dataset instead of the full LES class), and calculate some additional.

        Parameters
        ----------
        cbh_det_method: str
            Method to determine cloud base with:
                - if == "ql_cbh" then cbh is determined by a q_liq threshold set with the 'q_liq_cbh' attribute.
                - OTHER OPTIONS TO BE ADDED.
        """
        self.ds = self.ds.rename({self.height_dim: "height", self.time_dim: "time", self.pflux_field["name"]: "prec",
                        self.Ni_field["name"]: "Ni", self.T_field["name"]: "T", self.q_liq_field["name"]: "ql",
                        self.RH_field["name"]: "RH"})

        # scale and convert to float64
        for key in self.ds.keys():
            if self.ds[key].dtype == "float32":
                self.ds[key] = self.ds[key].astype(float)
        self.ds["RH"] *= self.RH_field["scaling"]
        self.ds["ql"] *= self.q_liq_field["scaling"]
        self.ds["T"] += self.T_field["addition"]
        self.ds["Ni"] *= self.Ni_field["scaling"]
        self.ds["prec"] *= self.pflux_field["scaling"]

        # set units
        self.ds["RH"].attrs["units"] = ""
        self.ds["ql"].attrs["units"] = "$g\: kg^{-1}$"
        self.ds["T"].attrs["units"] = "$K$"
        self.ds["Ni"].attrs["units"] = "$L^{-1}$"
        self.ds["prec"].attrs["units"] = "$mm\: h^{-1}$"

        # calculate ∆aw field for ABIFM
        self._calc_delta_aw()

        # calculated weighted precip rates
        self._find_and_calc_cb_precip(cbh_det_method)

    def _find_and_calc_cb_precip(self, cbh_det_method="ql_cbh"):
        """
        calculate number-weighted precip rate in the domain and allocate a field for values at lowest cloud base.

        Parameters
        ----------
        cbh_det_method: str
            Method to determine cloud base with:
                - if == "ql_cbh" then cbh is determined by a q_liq threshold set with the 'q_liq_cbh' attribute.
                - OTHER OPTIONS TO BE ADDED.
        """
        self.ds["P_Ni"] = self.ds['prec'] / self.ds['Ni']
        self.ds["P_Ni"].attrs['units'] = '$mm\: h^{-1}\: L^{-1}$'

        # find all cloud bases and the precip rate in the lowest cloud base in every time step (each profile).
        if cbh_det_method == "ql_cbh":
            self.ds["cbh_all"] = xr.DataArray(np.diff(self.ds["ql"].values >= self.q_liq_cbh, prepend=0,
                                                      axis=0) == 1, dims=self.ds["P_Ni"].dims)
        else:
            print("Unknown cbh method string - skipping cbh detection function")
            return
        self.ds["cbh_all"].attrs['long_name'] = "All detected cloud base heights (receive a 'True' value)"

        cbh_lowest = np.where(np.logical_and(np.cumsum(self.ds["cbh_all"], axis=0) == 1,
                                             self.ds["cbh_all"] == True))
        self.ds["lowest_cbh"] = xr.DataArray(np.zeros(self.ds.dims["time"]) * np.nan, dims=self.ds["time"].dims)
        self.ds["lowest_cbh"][cbh_lowest[1]] = self.ds["height"].values[cbh_lowest[0]]
        self.ds["lowest_cbh"].attrs['units'] = '$m$'
        self.ds["lowest_cbh"].attrs['long_name'] = "Lowest cloud base height per profile"
        self.ds["Pcb_per_Ni"] = xr.DataArray(np.zeros(self.ds.dims["time"]) * np.nan, dims=self.ds["time"].dims)
        self.ds["Pcb_per_Ni"][cbh_lowest[1]] = self.ds["P_Ni"].values[cbh_lowest]
        self.ds["Pcb_per_Ni"].attrs['units'] = '$mm\: h^{-1}\: L^{-1}$'
        self.ds["Pcb_per_Ni"].attrs['long_name'] = "Precipitation rate at the lowest cloud base per profile"

    def _calc_delta_aw(self):
        """
        calculate the ∆aw field for ABIFM
        """
        self.ds["delta_aw"] = self.ds['RH'] - \
                                (np.exp(9.550426-5723.265 / self.ds['T'] + 3.53068 * np.log(self.ds['T']) -
                                 0.00728332 * self.ds['T']) / ( np.exp(54.842763 - 6763.22 / self.ds['T'] -
                                 4.210 * np.log(self.ds['T']) + 0.000367 * self.ds['T'] + np.tanh(0.0415 *
                                 (self.ds['T'] - 218.8)) * (53.878 - 1331.22 / self.ds['T'] - 9.44523 *
                                 np.log(self.ds['T']) + 0.014025 * self.ds['T'])) ) )
        self.ds['delta_aw'].attrs['units'] = ""


class DHARMA(LES):
    def __init__(self, les_out_path=None, les_out_filename=None, t_harvest=None, fields_to_retain=None,
                 height_ind_2crop=None, cbh_det_method="ql_cbh"):
        """
        LES class for DHARMA that loads model output dataset

        Parameters
        ----------
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
        cbh_det_method: str
            Method to determine cloud base with:
                - if == "ql_cbh" then cbh is determined by a q_liq threshold set with the 'q_liq_cbh' attribute.
                - OTHER OPTIONS TO BE ADDED.

        """
        super().__init__()
        self.Ni_field = {"name": "ntot_3", "scaling": 1000}  # scale to L^-1
        self.pflux_field = {"name": "pflux_3", "scaling": 1}  # scale to mm/h
        self.T_field = {"name": "T", "addition": 0}  # scale to K (addition)
        self.q_liq_field = {"name": "ql", "scaling": 1000}  # scale to g/kg
        self.RH_field = {"name": "RH", "scaling": 1./100.}  # scale to fraction
        self.model_name = "DHARMA"
        self.time_dim = "time"
        self.height_dim = "zt"
        self.height_dim_2nd = "zw"

        # using the default ISDAC model output if None.
        if les_out_path is None:
            les_out_path = 'data_les/shi3_isdac_sfc6_pTqv_fthqv_lw_3_abifm_final/dharma.soundings.cdf'
        if les_out_filename is None:
            les_out_filename = 'dharma.soundings.cdf'

        # load model output
        self.ds = xr.open_dataset(les_out_path+les_out_filename)
        self.ds = self.ds.transpose(*(self.height_dim, self.time_dim, self.height_dim_2nd))  # make sure the height dim is on index 0 and make the 2nd height coordinates in the last dim.

        # crop specific model output time range (if requested)
        if t_harvest is not None:
            super()._crop_time_range(t_harvest)

        # crop specific model output fields and height range
        super()._crop_fields(fields_to_retain=fields_to_retain, height_ind_2crop=height_ind_2crop)

        # prepare fieldnames and generate new ones required for the 1D model
        super()._prepare_les_dataset_for_1d_model(cbh_det_method=cbh_det_method)
