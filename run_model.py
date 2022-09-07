"""
This module is used to run the model using a ci_model object.
"""
import xarray as xr
import numpy as np
from time import time


def run_model(ci_model):
    """
    Run the 1D model (output is stored in the ci_model object input to this method).
    Calculations are performed using numpy arrays rather than xarray DataArray indexing, because even though
    far less convenient and intuitive, numpy calculations have shown to be ~5-10 times faster than xarray (after
    optimization), which becomes significant with thousands of time steps.
    Note: calculations are performed in the following order (when ent_then_act is True item 2 comes prior to 1):
        1. aerosol activation since previous time step.
        2. Cloud-top entrainment of aerosol using entrainment rate from the previous time step.
        3. Turbulent mixing of aerosol using mixing depth and time scales from the previous time step (independent
           from the next steps).
        4. Ice sedimentation using ice fall velocity from the previous time step.
        5. Turbulent mixing of ice using mixing depth and time scales from the previous time step.

    Parameters
    ----------
    ci_model: ci_model class
        Containing variables such as the requested domain size, LES time averaging option
        (ci_model.t_averaged_les), custom or LES grid information (ci_model.custom_vert_grid),
        and LES xr.DataSet object(ci_model.les) after being processed.
        All these required data are automatically set when a ci_model class object is assigned
        during model initialization.
    """
    # Runtime stats
    Now = time()
    run_stats = {"activation_aer": 0, "entrainment_aer": 0, "mixing_aer": 0, "sedimentation_ice": 0,
                 "mixing_ice": 0, "data_allocation": 0}
    Complete_counter = 0
    print_delta_percent = 10.  # report model run progress every certain percentage of time steps.
    delta_t = ci_model.delta_t

    # budget output option
    if ci_model.output_budgets:
        if ci_model.do_mix_ice:
            ci_model.ds["budget_ice_mix"] = \
                xr.DataArray(np.zeros_like(ci_model.ds["T"]), dims=("height", "time"),
                             attrs={"units": "$m^{-3} s^{-1}$"})
        if ci_model.do_sedim:
            ci_model.ds["budget_ice_sedim"] = \
                xr.DataArray(np.zeros_like(ci_model.ds["T"]), dims=("height", "time"),
                             attrs={"units": "$m^{-3} s^{-1}$"})
        ci_model.ds["net_budget_0_test"] = \
            xr.DataArray(np.zeros_like(ci_model.ds["time"], dtype=float), dims=("time"),
                         attrs={"units": "$m^{-3}$"})

    # find indices of cloud-top height (based on ci_model.entrain_to_cth) for entrainment calculations
    use_cth_4_delta_n_calc = False  # if True, always use cth to calculate delta_N for entrainment.
    use_cth_delta_z = True  # if True, delta_z at cth for ent calc. Otherwise, delta_z = source - target
    cth_ind = np.argmin(np.abs(np.tile(np.expand_dims(ci_model.ds["lowest_cth"].values, axis=0),
                        (ci_model.mod_nz, 1)) - np.tile(np.expand_dims(ci_model.ds["height"], axis=1),
                                                        (1, ci_model.mod_nt))), axis=0)
    cth_ind = np.where(cth_ind == 0, -9999, cth_ind)
    mix_base_ind = np.argmin(np.abs(np.tile(np.expand_dims(ci_model.ds["mixing_base"].values, axis=0),
                                            (ci_model.mod_nz, 1)) -
                                    np.tile(np.expand_dims(ci_model.ds["height"], axis=1),
                                            (1, ci_model.mod_nt))), axis=0)

    ent_delta_z = [ci_model.ds["delta_z"].values[cth_ind[it]] for it in range(ci_model.mod_nt)]
    if isinstance(ci_model.entrain_to_cth, bool):
        if ci_model.entrain_to_cth:  # entrain to cth
            ent_target_ind = cth_ind
        else:  # entrain to PBL base (surface by default)
            ent_target_ind = mix_base_ind
            if not use_cth_delta_z:
                ent_delta_z = ci_model.ds["mixing_top"] - ci_model.ds["mixing_base"]
    elif isinstance(ci_model.entrain_to_cth, int):
        ent_target_ind = ci_model.entrain_to_cth
    if use_cth_4_delta_n_calc:
        ent_delta_n_ind = cth_ind
    else:
        ent_delta_n_ind = ent_target_ind  # index to use for delta_n calculation

    # init total INP arrays for INAS and subtract from n_aer (effective independent prognosed fields)
    for key in ci_model.aer.keys():
        if ci_model.prognostic_inp:
            if ci_model.aer[key].is_INAS:
                ci_model.aer[key].ds["inp_tot"] = xr.DataArray(np.zeros_like(ci_model.aer[key].ds["n_aer"]),
                                                               dims=("height", "time", "diam"))
                ci_model.aer[key].ds["inp_tot"][:, 0, :].values += \
                    ci_model.aer[key].ds["inp_snap"].copy().sum("T").values
                ci_model.aer[key].ds["n_aer"][:, 0, :].values -= \
                    ci_model.aer[key].ds["inp_tot"][:, 0, :].copy().values
                ci_model.aer[key].ds["inp_tot"].attrs["units"] = "$m^{-3}$"
                ci_model.aer[key].ds["inp_tot"].attrs["long_name"] = \
                    "Total prognosed INP subset number concentration"
            elif np.logical_and(not ci_model.use_ABIFM, not ci_model.aer[key].is_INAS):
                ci_model.aer[key].ds["n_aer"][:, 0].values -= \
                    ci_model.aer[key].ds["inp"][:, 0, :].copy().sum("T").values
        elif np.logical_and(ci_model.output_aer_decay, not ci_model.use_ABIFM):
            ci_model.aer[key].ds["pbl_inp_mean"] = \
                xr.DataArray(np.zeros(ci_model.ds["time"].shape), dims=(ci_model.time_dim))
            ci_model.aer[key].ds["pbl_inp_mean"].attrs["long_name"] = \
                "Mean PBL INP concentration (singular parameterization value)"
            ci_model.aer[key].ds["pbl_inp_mean"].attrs["units"] = "$m^{-3}$"

        # init budgets (if output option is True)
        if ci_model.output_budgets:
            if np.logical_or(ci_model.use_ABIFM,
                             np.logical_and(ci_model.aer[key].is_INAS, ci_model.prognostic_inp)):
                DIMS = ("height", "time", "diam")
                SHAPES = ci_model.aer[key].ds["n_aer"].shape
            else:
                DIMS = ("height", "time")
                SHAPES = ci_model.ds["T"].shape
            ci_model.aer[key].ds["budget_aer_act"] = \
                xr.DataArray(np.zeros(SHAPES), dims=DIMS,
                             attrs={"units": "$m^{-3} s^{-1}$"})
            if ci_model.do_entrain:
                ci_model.aer[key].ds["budget_aer_ent"] = \
                    xr.DataArray(np.zeros(ci_model.aer[key].ds["n_aer"].shape[1:]),
                                 dims=ci_model.aer[key].ds["n_aer"].dims[1:],
                                 attrs={"units": "$m^{-3} s^{-1}$"})
                if not np.logical_or(ci_model.use_ABIFM,
                                     np.logical_and(ci_model.aer[key].is_INAS, ci_model.prognostic_inp)):
                    budget_aer_ent = ci_model.aer[key].ds["budget_aer_ent"].values
            if ci_model.do_mix_aer:
                ci_model.aer[key].ds["budget_aer_mix"] = \
                    xr.DataArray(np.zeros(ci_model.aer[key].ds["n_aer"].shape),
                                 dims=ci_model.aer[key].ds["n_aer"].dims,
                                 attrs={"units": "$m^{-3} s^{-1}$"})

    if np.logical_and(ci_model.use_ABIFM, isinstance(ci_model.nuc_RH_thresh, float)):  # Define nucleation w/ RH
        in_cld_mask = ci_model.ds["RH"].values >= ci_model.nuc_RH_thresh
    elif np.logical_and(ci_model.use_ABIFM, isinstance(ci_model.nuc_RH_thresh, str)):
        if ci_model.nuc_RH_thresh == "use_ql":  # allow nucleation where ql > 0
            in_cld_mask = ci_model.ds["ql"].values >= ci_model.in_cld_q_thresh
    elif np.logical_and(ci_model.use_ABIFM, isinstance(ci_model.nuc_RH_thresh, list)):
        if ci_model.nuc_RH_thresh[0] == "use_RH_and_ql":  # allow nucleation where ql > 0 and/or RH > RH thresh
            in_cld_mask = np.logical_or(ci_model.ds["ql"].values >= ci_model.in_cld_q_thresh,
                                        ci_model.ds["RH"].values >= ci_model.nuc_RH_thresh[1])
    else:
        in_cld_mask = None

    for it in range(1, ci_model.mod_nt):

        t_loop = time()  # counter for a single loop
        t_proc = 0

        if it / ci_model.mod_nt * 100 > Complete_counter + print_delta_percent:
            Complete_counter += print_delta_percent
            print("%.0f%% of model run completed. Elapsed time: %.2f s." % (Complete_counter, time() - Now))

        n_ice_prev = np.zeros_like(ci_model.ds["Ni_nuc"].values[:, it - 1])
        n_ice_prev += ci_model.ds["Ni_nuc"].values[:, it - 1]
        n_ice_curr = ci_model.ds["Ni_nuc"].values[:, it]  # pointer for nucleated ice in current time step.
        n_ice_curr += n_ice_prev
        if ci_model.output_budgets:
            if ci_model.do_mix_ice:
                budget_ice_mix = ci_model.ds["budget_ice_mix"].values[:, it]
            if ci_model.do_sedim:
                budget_ice_sedim = ci_model.ds["budget_ice_sedim"].values[:, it]

        t_step_mix_mask = ci_model.ds["mixing_mask"].values[:, it - 1]  # mixed parts of profile = True

        if ci_model.time_splitting:
            n_ice_calc = n_ice_curr  # ptr
        else:
            n_ice_calc = n_ice_prev  # ptr

        for key in ci_model.aer.keys():
            if np.logical_or(ci_model.aer[key].is_INAS, ci_model.use_ABIFM):
                n_aer_prev = np.zeros_like(ci_model.aer[key].ds["n_aer"].values[:, it - 1, :])
                n_aer_prev += ci_model.aer[key].ds["n_aer"].values[:, it - 1, :]
                n_aer_curr = ci_model.aer[key].ds["n_aer"].values[:, it, :]  # ptr: aerosol conc. in current step.
                n_aer_curr += n_aer_prev
                diam_dim_l = ci_model.aer[key].ds["diam"].size
                if np.logical_and(ci_model.aer[key].is_INAS, ci_model.prognostic_inp):
                    if ci_model.output_budgets:
                        n_inp_prev_ref = np.zeros_like(ci_model.aer[key].ds["inp_snap"].values)
                        n_inp_prev_ref += ci_model.aer[key].ds["inp_snap"].values
                    n_inp_prev = np.zeros_like(ci_model.aer[key].ds["inp_snap"].values)
                    n_inp_prev += ci_model.aer[key].ds["inp_snap"].values
                    n_inp_curr = ci_model.aer[key].ds["inp_snap"].values  # ptr: INP conc. in current step.
                else:
                    n_inp_curr = None
                    n_inp_prev = None
            else:  # no aerosol diameter information under singular so n_aer_curr is a 1-D array (nz).
                n_aer_prev = np.zeros_like(ci_model.aer[key].ds["n_aer"].values[:, it - 1])
                n_aer_prev += ci_model.aer[key].ds["n_aer"].values[:, it - 1]
                n_aer_curr = ci_model.aer[key].ds["n_aer"].values[:, it]  # ptr: aerosol conc. in current step.
                n_aer_curr += n_aer_prev
                if ci_model.prognostic_inp:
                    n_inp_prev = np.zeros_like(ci_model.aer[key].ds["inp"].values[:, it - 1, :])
                    n_inp_prev += ci_model.aer[key].ds["inp"].values[:, it - 1, :]
                    n_inp_curr = ci_model.aer[key].ds["inp"].values[:, it, :]  # ptr: INP conc. in curr time step.
                    n_inp_curr += n_inp_prev
                else:
                    n_inp_curr = None
                    n_inp_prev = None
                diam_dim_l = None

            if ci_model.time_splitting:
                n_aer_calc = n_aer_curr  # ptr
                n_inp_calc = n_inp_curr
            else:
                n_aer_calc = n_aer_prev  # ptr
                n_inp_calc = n_inp_prev

            if ci_model.output_budgets:
                if np.logical_or(np.logical_and(ci_model.aer[key].is_INAS, ci_model.prognostic_inp),
                                 ci_model.use_ABIFM):
                    budget_aer_act = ci_model.aer[key].ds["budget_aer_act"].values[:, it, :]
                    if ci_model.do_entrain:
                        budget_aer_ent = ci_model.aer[key].ds["budget_aer_ent"].values[it, :]
                    if ci_model.do_mix_aer:
                        budget_aer_mix = ci_model.aer[key].ds["budget_aer_mix"].values[:, it, :]
                    inp_sum_dim = 2
                else:
                    budget_aer_act = ci_model.aer[key].ds["budget_aer_act"].values[:, it]
                    if ci_model.do_mix_aer:
                        budget_aer_mix = ci_model.aer[key].ds["budget_aer_mix"].values[:, it]
                    inp_sum_dim = 1
            else:
                inp_sum_dim = None
                budget_aer_act = None

            # Activate aerosol (ci_model.ent_then_act is False)
            if np.logical_and(ci_model.do_act, not ci_model.ent_then_act):
                t_process = time()
                n_aer_calc, n_inp_calc, budget_aer_act = \
                    activate_inp(ci_model, key, it, n_aer_calc, n_inp_calc, n_aer_curr, n_inp_curr, n_ice_curr,
                                 delta_t, budget_aer_act, inp_sum_dim, diam_dim_l, in_cld_mask)
                run_stats["activation_aer"] += (time() - t_process)

            # Cloud-top entrainment of aerosol and/or INP (depending on scheme).
            if ci_model.do_entrain:
                t_process = time()
                if np.logical_or(ci_model.aer[key].is_INAS, ci_model.use_ABIFM):  # aerosol entrainment
                    aer_ent = solve_entrainment(
                        ci_model.ds["w_e_ent"].values[it - 1], delta_t, ent_delta_z[it - 1],
                        ci_model.aer[key].ds["n_aer"].values[cth_ind[it - 1], 0, :],
                        n_aer_calc[ent_delta_n_ind[it - 1], :], ci_model.implicit_ent)
                    n_aer_curr[ent_target_ind[it - 1], :] += aer_ent
                    if ci_model.output_budgets:
                        budget_aer_ent += aer_ent / delta_t
                else:  # 1-D processing of n_aer_curr for singular.
                    aer_ent = solve_entrainment(
                        ci_model.ds["w_e_ent"].values[it - 1], delta_t, ent_delta_z[it - 1],
                        ci_model.aer[key].ds["n_aer"].values[cth_ind[it - 1], 0],
                        n_aer_calc[ent_delta_n_ind[it - 1]], ci_model.implicit_ent)
                    n_aer_curr[ent_target_ind[it - 1]] += aer_ent
                    if ci_model.output_budgets:
                        budget_aer_ent[it] += aer_ent / delta_t
                if np.logical_and(not ci_model.use_ABIFM, ci_model.prognostic_inp):  # INP entrainment
                    if ci_model.aer[key].is_INAS:  # additional dim (diam) for INAS
                        inp_ent = solve_entrainment(
                            ci_model.ds["w_e_ent"].values[it - 1], delta_t, ent_delta_z[it - 1],
                            ci_model.aer[key].ds["inp_init"].values[cth_ind[it - 1], :, :],
                            n_inp_calc[ent_delta_n_ind[it - 1], :, :], ci_model.implicit_ent)
                        n_inp_curr[ent_target_ind[it - 1], :, :] += inp_ent
                        if ci_model.output_budgets:
                            budget_aer_ent += inp_ent.sum(axis=inp_sum_dim - 1) / delta_t
                    else:
                        inp_ent = solve_entrainment(
                            ci_model.ds["w_e_ent"].values[it - 1], delta_t, ent_delta_z[it - 1],
                            ci_model.aer[key].ds["inp"].values[cth_ind[it - 1], 0, :],
                            n_inp_calc[ent_delta_n_ind[it - 1], :], ci_model.implicit_ent)
                        n_inp_curr[ent_target_ind[it - 1], :] += inp_ent
                        if ci_model.output_budgets:
                            budget_aer_ent[it] += inp_ent.sum(axis=inp_sum_dim - 1) / delta_t
                run_stats["entrainment_aer"] += (time() - t_process)
                t_proc += time() - t_process

            # Activate aerosol (ci_model.ent_then_act is True)
            if np.logical_and(ci_model.do_act, ci_model.ent_then_act):
                t_process = time()
                n_aer_calc, n_inp_calc, budget_aer_act = \
                    activate_inp(ci_model, key, it, n_aer_calc, n_inp_calc, n_aer_curr, n_inp_curr, n_ice_curr,
                                 delta_t, budget_aer_act, inp_sum_dim, diam_dim_l, in_cld_mask)
                run_stats["activation_aer"] += (time() - t_process)

            # Turbulent mixing of aerosol
            if ci_model.do_mix_aer:
                t_process = time()
                if np.any(t_step_mix_mask):  # checking that some mixing takes place
                    if np.logical_or(ci_model.aer[key].is_INAS, ci_model.use_ABIFM):  # aerosol mixing
                        if np.all(t_step_mix_mask):  # Faster processing for fully mixed domain
                            aer_fully_mixed = np.nanmean(n_aer_calc, axis=0)
                            aer_mixing = delta_t / ci_model.ds["tau_mix"].values[it - 1] * \
                                (np.tile(np.expand_dims(aer_fully_mixed, axis=0), (ci_model.mod_nz, 1)) -
                                 n_aer_calc)
                            n_aer_curr += aer_mixing
                            if ci_model.output_budgets:
                                budget_aer_mix += aer_mixing / delta_t

                        else:
                            aer_fully_mixed = np.nanmean(np.where(np.tile(np.expand_dims(t_step_mix_mask, axis=1),
                                                                          (1, diam_dim_l)),
                                                                  n_aer_calc, np.nan), axis=0)
                            aer_mixing = delta_t / ci_model.ds["tau_mix"].values[it - 1] * \
                                (np.tile(np.expand_dims(aer_fully_mixed, axis=0), (np.sum(t_step_mix_mask), 1)) -
                                 n_aer_calc[t_step_mix_mask, :])
                            n_aer_curr[t_step_mix_mask, :] += aer_mixing
                            if ci_model.output_budgets:
                                budget_aer_mix[t_step_mix_mask, :] += aer_mixing / delta_t
                    else:
                        if np.all(t_step_mix_mask):  # Faster processing for fully mixed domain
                            aer_fully_mixed = np.nanmean(n_aer_calc)
                            aer_mixing = delta_t / ci_model.ds["tau_mix"].values[it - 1] * \
                                (aer_fully_mixed - n_aer_calc)
                            n_aer_curr += aer_mixing
                            if ci_model.output_budgets:
                                budget_aer_mix += aer_mixing / delta_t
                        else:
                            aer_fully_mixed = np.nanmean(np.where(t_step_mix_mask, n_aer_calc, np.nan))
                            aer_mixing = delta_t / ci_model.ds["tau_mix"].values[it - 1] * \
                                (aer_fully_mixed - n_aer_calc[t_step_mix_mask])
                            n_aer_curr[t_step_mix_mask] += aer_mixing
                            if ci_model.output_budgets:
                                budget_aer_mix[t_step_mix_mask] += aer_mixing / delta_t
                if np.logical_and(not ci_model.use_ABIFM, ci_model.prognostic_inp):  # INP mixing
                    if np.any(t_step_mix_mask):  # checking that some mixing takes place.
                        if np.all(t_step_mix_mask):  # Faster processing for fully mixed domain
                            inp_fully_mixed = np.nanmean(n_inp_calc, axis=0)
                            if ci_model.aer[key].is_INAS:  # additional dim (diam) for INAS
                                inp_mixing = delta_t / ci_model.ds["tau_mix"].values[it - 1] * \
                                    (np.tile(np.expand_dims(inp_fully_mixed, axis=0), (ci_model.mod_nz, 1, 1)) -
                                     n_inp_calc)
                            else:
                                inp_mixing = delta_t / ci_model.ds["tau_mix"].values[it - 1] * \
                                    (np.tile(np.expand_dims(inp_fully_mixed, axis=0), (ci_model.mod_nz, 1)) -
                                     n_inp_calc)
                            n_inp_curr += inp_mixing
                            if ci_model.output_budgets:
                                budget_aer_mix += inp_mixing.sum(axis=inp_sum_dim) / delta_t
                        else:
                            if ci_model.aer[key].is_INAS:  # additional dim (diam) for INAS
                                inp_fully_mixed = np.nanmean(np.where(np.tile(np.expand_dims(t_step_mix_mask,
                                                                                             axis=(1, 2)),
                                                                              (1, diam_dim_l,
                                                                               ci_model.aer[key].ds["T"].size)),
                                                                      n_inp_calc, np.nan), axis=0)
                                inp_mixing = delta_t / ci_model.ds["tau_mix"].values[it - 1] * \
                                    (np.tile(np.expand_dims(inp_fully_mixed, axis=0),
                                             (np.sum(t_step_mix_mask), 1, 1)) - n_inp_calc[t_step_mix_mask, :, :])
                                n_inp_curr[t_step_mix_mask, :, :] += inp_mixing
                                if ci_model.output_budgets:
                                    budget_aer_mix[t_step_mix_mask, :] += \
                                        inp_mixing.sum(axis=inp_sum_dim) / delta_t
                            else:
                                inp_fully_mixed = np.nanmean(np.where(np.tile(np.expand_dims(t_step_mix_mask,
                                                                                             axis=1),
                                                                              (1, ci_model.aer[key].ds["T"].size)),
                                                                      n_inp_calc, np.nan), axis=0)
                                inp_mixing = delta_t / ci_model.ds["tau_mix"].values[it - 1] * \
                                    (np.tile(np.expand_dims(inp_fully_mixed, axis=0),
                                             (np.sum(t_step_mix_mask), 1)) - n_inp_calc[t_step_mix_mask, :])
                                n_inp_curr[t_step_mix_mask, :] += inp_mixing
                                if ci_model.output_budgets:
                                    budget_aer_mix[t_step_mix_mask] += \
                                        inp_mixing.sum(axis=inp_sum_dim) / delta_t
                run_stats["mixing_aer"] += (time() - t_process)
                t_proc += time() - t_process

            # Place resolved aerosol and/or INP
            place_resolved_aer(ci_model, key, it, n_aer_curr, n_inp_curr)
            if ci_model.output_budgets:
                if ci_model.use_ABIFM:
                    ci_model.ds["net_budget_0_test"].values[it] += \
                        (ci_model.aer[key].ds["n_aer"].values[:, it - 1, :] -
                         n_aer_curr).sum()
                elif ci_model.aer[key].is_INAS:
                    if ci_model.prognostic_inp:
                        ci_model.ds["net_budget_0_test"].values[it] += \
                            (ci_model.aer[key].ds["n_aer"].values[:, it - 1, :] - n_aer_curr +
                             (n_inp_prev_ref - n_inp_curr).sum(axis=-1)).sum()
                    else:
                        ci_model.ds["net_budget_0_test"].values[it] += \
                            (ci_model.aer[key].ds["n_aer"].values[:, it - 1, :] - n_aer_curr).sum()
                else:
                    ci_model.ds["net_budget_0_test"].values[it] += \
                        (ci_model.aer[key].ds["n_aer"].values[:, it - 1] - n_aer_curr).sum()
                    if ci_model.prognostic_inp:
                        ci_model.ds["net_budget_0_test"].values[it] += \
                            (ci_model.aer[key].ds["inp"].values[:, it - 1, :] - n_inp_curr).sum()
                if ci_model.do_entrain:
                    if ci_model.use_ABIFM:
                        ci_model.ds["net_budget_0_test"].values[it] += aer_ent.sum()
                    elif ci_model.prognostic_inp:
                        ci_model.ds["net_budget_0_test"].values[it] += (aer_ent + inp_ent.sum(axis=-1)).sum()

        # Sedimentation of ice (after aerosol were activated).
        if ci_model.do_sedim:
            t_process = time()
            if ci_model.ds["v_f_ice"].ndim == 2:
                ice_sedim_out = n_ice_prev * ci_model.ds["v_f_ice"].values[:, it - 1] * delta_t / \
                    ci_model.ds["delta_z"].values
            else:
                ice_sedim_out = n_ice_prev * ci_model.ds["v_f_ice"].values[it - 1] * delta_t / \
                    ci_model.ds["delta_z"].values
            ice_sedim_in = np.zeros(ci_model.mod_nz)
            ice_sedim_in[:-1] += ice_sedim_out[1:]
            n_ice_curr += ice_sedim_in
            n_ice_curr -= ice_sedim_out
            if ci_model.output_budgets:
                budget_ice_sedim += (ice_sedim_in - ice_sedim_out) / delta_t
            t_proc += time() - t_process
            run_stats["sedimentation_ice"] += (time() - t_process)

        # Turbulent mixing of ice
        if ci_model.do_mix_ice:
            t_process = time()
            if np.any(t_step_mix_mask):  # checking that some mixing takes place.
                if np.all(t_step_mix_mask):  # Faster processing for fully mixed domain
                    ice_fully_mixed = np.nanmean(n_ice_calc, axis=0)
                    ice_mixing = delta_t / ci_model.ds["tau_mix"].values[it - 1] * \
                        (ice_fully_mixed - n_ice_calc)
                    n_ice_curr += ice_mixing
                    if ci_model.output_budgets:
                        budget_ice_mix += ice_mixing / delta_t
                else:
                    ice_fully_mixed = np.nanmean(np.where(t_step_mix_mask, n_ice_calc, np.nan), axis=0)
                    ice_mixing = delta_t / ci_model.ds["tau_mix"].values[it - 1] * \
                        (ice_fully_mixed - n_ice_calc[t_step_mix_mask])
                    n_ice_curr[t_step_mix_mask] += ice_mixing
                    if ci_model.output_budgets:
                        budget_ice_mix[t_step_mix_mask] += ice_mixing / delta_t
            t_proc += time() - t_process
            run_stats["mixing_ice"] += (time() - t_process)

        # Place resolved ice
        ci_model.ds["Ni_nuc"][:, it].values = n_ice_curr
        if ci_model.output_budgets:
            ci_model.ds["net_budget_0_test"].values[it] += \
                (n_ice_prev - n_ice_curr).sum() - ice_sedim_out[0].sum()

        run_stats["data_allocation"] += (time() - t_loop - t_proc)

    # Reassign units (often occurring in xarray in data allocation) & and add total INP to n_aero in INAS
    for key in ci_model.aer.keys():
        if 'diam' in ci_model.aer[key].ds.keys():
            if "units" not in ci_model.aer[key].ds['diam'].attrs:
                ci_model.aer[key].ds["diam"].attrs["units"] = r"$\mu m$"
        if 'T' in ci_model.aer[key].ds.keys():
            if "units" not in ci_model.aer[key].ds['T'].attrs:
                ci_model.aer[key].ds["T"].attrs["units"] = r"$K$"
        if "units" not in ci_model.aer[key].ds['time'].attrs:
            ci_model.aer[key].ds["time"].attrs["units"] = r"$s$"
        if "units" not in ci_model.aer[key].ds['height'].attrs:
            ci_model.aer[key].ds["height"].attrs["units"] = r"$m$"

        if ci_model.prognostic_inp:
            if ci_model.aer[key].is_INAS:
                ci_model.aer[key].ds["n_aer"].values += ci_model.aer[key].ds["inp_tot"].values
                sum_dims = (ci_model.height_dim, ci_model.diam_dim)  # for aer decay calcs if output_aer_decay
            elif np.logical_and(not ci_model.use_ABIFM, not ci_model.aer[key].is_INAS):
                ci_model.aer[key].ds["n_aer"].values += ci_model.aer[key].ds["inp"].copy().sum("T").values
                sum_dims = (ci_model.height_dim)  # for aer decay calcs if output_aer_decay
            else:
                sum_dims = (ci_model.height_dim, ci_model.diam_dim)  # for aer decay calcs if output_aer_decay

            # Calculate INP and aerosol decay statistics
            if ci_model.output_aer_decay:
                ci_model.aer[key].ds["pbl_aer_tot_rel_frac"] = \
                    xr.DataArray((ci_model.aer[key].ds["n_aer"] * ci_model.ds["delta_z"] *
                                  ci_model.ds["mixing_mask"]).sum(sum_dims) /
                                 (ci_model.aer[key].ds["n_aer"].isel({ci_model.time_dim: 0}) *
                                  ci_model.ds["delta_z"] * ci_model.ds["mixing_mask"]).sum(sum_dims),
                                 dims=(ci_model.time_dim),
                                 attrs={"long_name": "Fraction of total PBL aerosol relative to initial"})
                ci_model.aer[key].ds["pbl_aer_tot_decay_rate"] = \
                    xr.DataArray(np.diff((ci_model.aer[key].ds["n_aer"] * ci_model.ds["delta_z"] *
                                         ci_model.ds["mixing_mask"]).sum(sum_dims),
                                         prepend=np.nan) / delta_t * (-1),
                                 dims=(ci_model.time_dim),
                                 attrs={"units": "$m^{-2} s^{-1}$",
                                        "long_name": "Total PBL aerosol decay rate (multiplied by -1)"})
                if ci_model.use_ABIFM:
                    ci_model.aer[key].ds["pbl_inp_tot_rel_frac"] = \
                        xr.DataArray(ci_model.aer[key].ds["pbl_aer_tot_rel_frac"].values,
                                     dims=(ci_model.time_dim))
                    ci_model.aer[key].ds["pbl_inp_mean"] = \
                        xr.DataArray((ci_model.aer[key].ds["n_aer"] *
                                      ci_model.ds["mixing_mask"]).sum(sum_dims) /
                                     ci_model.ds["mixing_mask"].sum(axis=0),
                                     dims=(ci_model.time_dim))
                elif ci_model.aer[key].is_INAS:
                    ci_model.aer[key].ds["pbl_inp_tot_rel_frac"] = \
                        xr.DataArray((ci_model.aer[key].ds["inp_tot"] * ci_model.ds["delta_z"] *
                                      ci_model.ds["mixing_mask"]).sum(sum_dims) /
                                     (ci_model.aer[key].ds["inp_tot"].isel({ci_model.time_dim: 0}) *
                                      ci_model.ds["delta_z"] * ci_model.ds["mixing_mask"]).sum(sum_dims),
                                     dims=(ci_model.time_dim))
                    ci_model.aer[key].ds["pbl_inp_mean"] = \
                        xr.DataArray((ci_model.aer[key].ds["inp_tot"] *
                                      ci_model.ds["mixing_mask"]).sum(sum_dims) /
                                     ci_model.ds["mixing_mask"].sum(axis=0),
                                     dims=(ci_model.time_dim))
                else:
                    ci_model.aer[key].ds["pbl_inp_tot_rel_frac"] = \
                        xr.DataArray((ci_model.aer[key].ds["inp"] * ci_model.ds["delta_z"] *
                                      ci_model.ds["mixing_mask"]).sum((ci_model.height_dim, ci_model.T_dim)) /
                                     (ci_model.aer[key].ds["inp"].isel({ci_model.time_dim: 0}) *
                                      ci_model.ds["delta_z"] *
                                      ci_model.ds["mixing_mask"]).sum((ci_model.height_dim, ci_model.T_dim)),
                                     dims=(ci_model.time_dim))
                    ci_model.aer[key].ds["pbl_inp_mean"] = \
                        xr.DataArray((ci_model.aer[key].ds["inp"] *
                                      ci_model.ds["mixing_mask"]).sum((ci_model.height_dim, ci_model.T_dim)) /
                                     ci_model.ds["mixing_mask"].sum(axis=0),
                                     dims=(ci_model.time_dim))
                ci_model.aer[key].ds["pbl_inp_tot_rel_frac"].attrs["long_name"] = \
                    "Fraction of total PBL activatable INP relative to initial"
                ci_model.aer[key].ds["pbl_inp_mean"].attrs["long_name"] = "Mean PBL INP concentration"
                ci_model.aer[key].ds["pbl_inp_mean"].attrs["units"] = "$m^{-3}$"
        elif np.logical_and(ci_model.output_aer_decay, not ci_model.use_ABIFM):
            ci_model.aer[key].ds["pbl_inp_mean"].values[0] += ci_model.aer[key].ds["pbl_inp_mean"].values[1]
    ci_model.ds["pbl_ice_mean"] = \
        xr.DataArray((ci_model.ds["Ni_nuc"] *
                      ci_model.ds["mixing_mask"]).sum("height") /
                     ci_model.ds["mixing_mask"].sum("height"),
                     dims=(ci_model.time_dim))
    ci_model.ds["pbl_ice_mean"].attrs["long_name"] = "Mean PBL ice concentration"
    ci_model.ds["pbl_ice_mean"].attrs["units"] = "$m^{-3}$"

    # Finalize arrays
    for key in ci_model.aer.keys():
        for DA in ci_model.aer[key].ds.keys():
            if "units" in ci_model.aer[key].ds[DA].attrs:
                ci_model.aer[key].ds[DA].data = ci_model.aer[key].ds[DA].data * \
                    ci_model.ureg(ci_model.aer[key].ds[DA].attrs["units"])
    for DA in ci_model.ds.keys():
        if "units" in ci_model.ds[DA].attrs:
            ci_model.ds[DA].data = ci_model.ds[DA].data * ci_model.ureg(ci_model.ds[DA].attrs["units"])

    # Print model run summary
    runtime_tot = time() - Now
    print("\nModel run finished! Total run time = %f s\nModel run time stats:" % runtime_tot)
    for key in run_stats.keys():
        print("Process: %s: %.2f s (%.2f%% of of total time)" %
              (key, run_stats[key], run_stats[key] / runtime_tot * 100.))
    print("\n")


def activate_inp(ci_model, key, it, n_aer_calc, n_inp_calc, n_aer_curr, n_inp_curr, n_ice_curr, delta_t,
                 budget_aer_act=None, inp_sum_dim=None, diam_dim_l=None, in_cld_mask=None):
    """
    Activate INP based using the appropriate method (note that n_aer_curr and n_inp_curr are not returned
    as these are either None or pointers, as in the case of the ci_model object as well).

    Parameters
    ----------
    ci_model: ci_model class
        Containing variables such as the requested domain size, LES time averaging option
        (ci_model.t_averaged_les), custom or LES grid information (ci_model.custom_vert_grid),
        and LES xr.DataSet object(ci_model.les) after being processed.
        All these required data are automatically set when a ci_model class object is assigned
        during model initialization.
    key: str
        aerosol population name.
    it: int
        time step index.
    n_aer_calc: np.ndarray
        array containing aerosol data used for prognostic calculations.
    n_inp_calc: np.ndarray
        array containing INP data used for prognostic calculations.
    n_aer_curr: np.ndarray
        array containing current aerosol data to be updated.
    n_inp_curr: np.ndarray
        array containing current INP data to be updated.
    n_ice_curr: np.ndarray
        array containing current ice data to be updated.
    delta_t: float
        time_step [s].
    budget_aer_act: np.ndarray
        activation budget array.
    inp_sum_dim: int
        dimension to sum up when calculating total activated.
    diam_dim_l: int
        length of diameter dimension in the aer array [--INAS--or--ABIFM--].
    in_cld_mask: np.ndarray of bool
        array masking cloudy grid cells wherein activation takes place [--ABIFM--]

    Returns
    -------
    n_aer_calc: np.ndarray
        array containing aerosol data used for prognostic calculations.
    n_inp_calc: np.ndarray
        array containing INP data used for prognostic calculations.
    budget_aer_act: np.ndarray
        activation budget array.
    """
    if ci_model.use_ABIFM:
        AA, JJ = np.meshgrid(ci_model.aer[key].ds["surf_area"].values,
                             ci_model.aer[key].ds["Jhet"].values[:, it - 1])
        aer_act = np.minimum(n_aer_calc * JJ * AA * delta_t, n_aer_calc)
        if not ci_model.prognostic_inp:
            tmp_ice_curr = np.tile(np.expand_dims(n_ice_curr, axis=1), (1, diam_dim_l))
            aer_act = np.where(aer_act < tmp_ice_curr, 0., aer_act - tmp_ice_curr)  # aer_act is max using Ninp + Nice
        if ci_model.nuc_RH_thresh is not None:
            aer_act = np.where(np.tile(np.expand_dims(in_cld_mask[:, it - 1], axis=1), (1, diam_dim_l)),
                               aer_act, 0.)
    else:
        if ci_model.prognostic_inp:
            TTi, TTm = np.meshgrid(ci_model.aer[key].ds["T"].values,
                                   np.where(ci_model.ds["ql"].values[:, it - 1] >= ci_model.in_cld_q_thresh,
                                            ci_model.ds["T"].values[:, it - 1], np.nan))  # ignore noncld cells
            if ci_model.aer[key].is_INAS:
                aer_act = np.minimum(np.where(np.tile(np.expand_dims(TTi >= TTm, axis=1), (1, diam_dim_l, 1)),
                                              n_inp_calc, 0), n_inp_calc)
            else:
                aer_act = np.minimum(np.where(TTi >= TTm, n_inp_calc, 0), n_inp_calc)
        else:
            if ci_model.aer[key].is_INAS:
                aer_act = ci_model.aer[key].singular_fun(
                    ci_model.ds["T"].values[:, it - 1],
                    (np.tile(np.expand_dims(ci_model.aer[key].ds["surf_area"].values, axis=0),
                             (ci_model.ds["height"].size, 1)) * n_aer_calc)).sum(axis=1)
            else:
                aer_act = ci_model.aer[key].singular_fun(ci_model.ds["T"].values[:, it - 1],
                                                         n_aer_calc * ci_model.aer[key].n_aer05_frac)
            if ci_model.aer[key].singular_scale != 1.:  # scale INP option
                aer_act *= ci_model.singular_scale
            if ci_model.output_aer_decay:
                ci_model.aer[key].ds["pbl_inp_mean"][it] += \
                    (aer_act * ci_model.ds["mixing_mask"].values[:, it]).max()  # simply the PBL maximum
            aer_act = np.where(aer_act < n_ice_curr, 0., aer_act - n_ice_curr)  # aer_act is max using Ninp + Nice
            aer_act = np.where(ci_model.ds["ql"].values[:, it - 1] >= ci_model.in_cld_q_thresh,
                               aer_act, 0.)
        if ci_model.use_tau_act:
            if ci_model.implicit_act:
                aer_act = aer_act - aer_act / (1 + delta_t / ci_model.tau_act)  # n(t) - n(t+1)
            elif delta_t < ci_model.tau_act:  # explicit (can't make aer_act larger when dt > tau_act)
                aer_act = aer_act * delta_t / ci_model.tau_act
    if ci_model.prognostic_inp:
        if ci_model.aer[key].is_INAS:
            ci_model.ds["nuc_rate"].values[:, it] += aer_act.sum(axis=(1, 2)) / delta_t  # nuc. rate
        else:
            ci_model.ds["nuc_rate"].values[:, it] += aer_act.sum(axis=1) / delta_t  # nucleation rate
    elif ci_model.use_ABIFM:
        ci_model.ds["nuc_rate"].values[:, it] += aer_act.sum(axis=1) / delta_t  # nucleation rate
    else:
        ci_model.ds["nuc_rate"].values[:, it] += aer_act / delta_t  # nucleation rate
    if ci_model.use_ABIFM:
        if ci_model.prognostic_inp:
            n_aer_curr -= aer_act  # Subtract from aerosol reservoir.
        n_ice_curr += aer_act.sum(axis=1)  # Add from aerosol reservoir (*currently*, w/o aerosol memory)
        if ci_model.output_budgets:
            budget_aer_act -= aer_act / delta_t
    else:
        if ci_model.prognostic_inp:
            n_inp_curr -= aer_act  # Subtract from aerosol reservoir (INP subset).
            if ci_model.aer[key].is_INAS:
                n_ice_curr += aer_act.sum(axis=(1, 2))  # Add from reservoir (*currently*, w/o aerosol memory)
            else:
                n_ice_curr += aer_act.sum(axis=1)  # Add from reservoir (*currently*, w/o aerosol memory)
            if ci_model.output_budgets:
                if ci_model.prognostic_inp:
                    budget_aer_act -= aer_act.sum(axis=inp_sum_dim) / delta_t
        else:
            n_ice_curr += aer_act
            if ci_model.output_budgets:
                budget_aer_act -= aer_act / delta_t
    return n_aer_calc, n_inp_calc, budget_aer_act


def solve_entrainment(w_e_ent, delta_t, delta_z, n_ft, n_pblh, implicit_solver=True):
    """
    Calculate entrainment source using implicit or explicit method.

    Parameters
    ----------
    w_e_ent: float or int
        PBL top entrainment rate [m/s].
    delta_t: float
        time_step [s].
    delta_z: float
        vertical spacing at PBL (cloud) top
    n_ft: float or np.ndarray
        Free-troposphere (or source) aerosol (or INP) concentration.
    n_pblh: float or np.ndarray
        PBL top (or target level) aerosol (or INP) concentration.
    implicit_solver: bool
        True - using implicit solver, False - explicit solver.

    Returns
    -------
    ent_src: float or np.ndarray
        entrainment source.
    """
    if implicit_solver:
        ent_src = (n_pblh + w_e_ent / delta_z * delta_t * n_ft) / \
            (1 + w_e_ent / delta_z * delta_t) - n_pblh  # n(t+1) - n(t)
    else:
        ent_src = w_e_ent / delta_z * delta_t * (n_ft - n_pblh)
    return ent_src


def place_resolved_aer(ci_model, key, it, n_aer_curr, n_inp_curr=None):
    """
    Place resolved aerosol and/or INP in the ci_model object.

    Parameters
    ----------
    ci_model: ci_model class
        Containing variables such as the requested domain size, LES time averaging option
        (ci_model.t_averaged_les), custom or LES grid information (ci_model.custom_vert_grid),
        and LES xr.DataSet object(ci_model.les) after being processed.
        All these required data are automatically set when a ci_model class object is assigned
        during model initialization.
    key: str
        aerosol population name
    it: int
        time step index
    n_aer_curr: np.ndarray
        array of current aerosol state
    n_inp_curr: np.ndarray (singular) or None (ABIFM)
        array of current INP state
    """
    if ci_model.use_ABIFM:
        ci_model.aer[key].ds["n_aer"][:, it, :].values = n_aer_curr
    elif ci_model.aer[key].is_INAS:
        ci_model.aer[key].ds["n_aer"][:, it, :].values = n_aer_curr
        if ci_model.prognostic_inp:
            ci_model.aer[key].ds["inp_tot"][:, it, :].values += np.sum(n_inp_curr, axis=2)
    else:
        if ci_model.prognostic_inp:
            ci_model.aer[key].ds["inp"].values[:, it, :] = n_inp_curr
        ci_model.aer[key].ds["n_aer"][:, it].values = n_aer_curr
