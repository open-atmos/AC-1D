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
    Note: calculations are performed in the following order (mass is conserved):
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

    # find indices of cloud-top height (based on ci_model.entrain_to_cth) for entrainment calculations
    use_cth_4_delta_n_calc = False  # if True, always use cth to calculate delta_N for entrainment.
    use_cth_delta_z = True  # if True, delta_z at cth for ent calc. Otherwise, delta_z = source - target
    cth_ind = np.argmin(np.abs(np.tile(np.expand_dims(ci_model.ds["lowest_cth"].values, axis=0),
                        (ci_model.mod_nz, 1)) - np.tile(np.expand_dims(ci_model.ds["height"], axis=1),
                                                        (1, ci_model.mod_nt))), axis=0)
    cth_ind = np.where(cth_ind == 0, -9999, cth_ind)
    ent_delta_z = [ci_model.ds["delta_z"].values[cth_ind[it]] for it in range(ci_model.mod_nt)]
    if ci_model.entrain_to_cth:  # entrain to cth
        ent_target_ind = cth_ind
    else:  # entrain to PBL base (surface by default)
        ent_target_ind = np.argmin(np.abs(np.tile(np.expand_dims(ci_model.ds["mixing_base"].values, axis=0),
                                   (ci_model.mod_nz, 1)) - np.tile(np.expand_dims(ci_model.ds["height"], axis=1),
                                                                   (1, ci_model.mod_nt))), axis=0)
        if not use_cth_delta_z:
            ent_delta_z = ci_model.ds["mixing_top"] - ci_model.ds["mixing_base"]
    if use_cth_4_delta_n_calc:
        ent_delta_n_ind = cth_ind
    else:
        ent_delta_n_ind = ent_target_ind  # index to use for delta_n calculation

    # init total INP arrays for INAS and subtract from n_aer (effective independent prognosed fields)
    for key in ci_model.aer.keys():
        if ci_model.aer[key].is_INAS:
            ci_model.aer[key].ds["inp_tot"] = xr.DataArray(np.zeros_like(ci_model.aer[key].ds["n_aer"]),
                                                           dims=("height", "time", "diam"))
            ci_model.aer[key].ds["inp_tot"][:, 0, :].values += \
                ci_model.aer[key].ds["inp_snap"].copy().sum("T").values
            ci_model.aer[key].ds["n_aer"][:, 0, :].values -= \
                ci_model.aer[key].ds["inp_tot"][:, 0, :].copy().values
            ci_model.aer[key].ds["inp_tot"].attrs["units"] = "$m^{-3}$"
            ci_model.aer[key].ds["inp_tot"].attrs["long_name"] = "Total prognosed INP subset number concentration"
        elif np.logical_and(not ci_model.use_ABIFM, not ci_model.aer[key].is_INAS):
            ci_model.aer[key].ds["n_aer"][:, 0].values -= \
                ci_model.aer[key].ds["inp"][:, 0, :].copy().sum("T").values

        # init budgets (if output option is True)
        if ci_model.output_budgets:
            if np.logical_or(ci_model.use_ABIFM, ci_model.aer[key].is_INAS):
                DIMS = ("height", "time", "diam")
            else:
                DIMS = ("height", "time") 
            ci_model.aer[key].ds["budget_aer_act"] = \
                xr.DataArray(np.zeros_like(ci_model.aer[key].ds["n_aer"]), dims=DIMS,
                             attrs={"units": "$m^{-3} s^{-1}$"})
            if ci_model.do_entrain:
                ci_model.aer[key].ds["budget_aer_ent"] = \
                    xr.DataArray(np.zeros(ci_model.aer[key].ds["n_aer"].shape[1:]), dims=DIMS[1:],
                                 attrs={"units": "$m^{-3} s^{-1}$"})
                if not np.logical_or(ci_model.use_ABIFM, ci_model.aer[key].is_INAS):
                    budget_aer_ent = ci_model.aer[key].ds["budget_aer_ent"].values
            if ci_model.do_mix_aer:
                ci_model.aer[key].ds["budget_aer_mix"] = \
                    xr.DataArray(np.zeros_like(ci_model.aer[key].ds["n_aer"]), dims=DIMS,
                                 attrs={"units": "$m^{-3} s^{-1}$"})

    if np.logical_and(ci_model.use_ABIFM, isinstance(ci_model.nuc_RH_thresh, float)):  # Define nucleation w/ RH
        in_cld_mask = ci_model.ds["RH"] >= ci_model.nuc_RH_thresh
    elif np.logical_and(ci_model.use_ABIFM, isinstance(ci_model.nuc_RH_thresh, str)):
        if ci_model.nuc_RH_thresh == "use_ql":  # allow nucleation where ql > 0
            in_cld_mask = ci_model.ds["ql"] >= ci_model.in_cld_q_thresh
    elif np.logical_and(ci_model.use_ABIFM, isinstance(ci_model.nuc_RH_thresh, list)):
        if ci_model.nuc_RH_thresh[0] == "use_RH_and_ql":  # allow nucleation where ql > 0 and/or RH > RH thresh
            in_cld_mask = np.logical_or(ci_model.ds["ql"] >= ci_model.in_cld_q_thresh,
                                        ci_model.ds["RH"] >= ci_model.nuc_RH_thresh[1])

    for it in range(1, ci_model.mod_nt):

        t_loop = time()  # counter for a single loop
        t_proc = 0

        if it / ci_model.mod_nt * 100 > Complete_counter + print_delta_percent:
            Complete_counter += print_delta_percent
            print("%.0f%% of model run completed. Elapsed time: %.2f s." % (Complete_counter, time() - Now))

        n_ice_prev = ci_model.ds["Ni_nuc"].values[:, it - 1]  # pointer for nucleated ice in prev. time step.
        n_ice_curr = ci_model.ds["Ni_nuc"].values[:, it]  # pointer for nucleated ice in current time step.
        n_ice_curr += n_ice_prev
        if ci_model.output_budgets:
            if ci_model.do_mix_ice:
                budget_ice_mix = ci_model.ds["budget_ice_mix"].values[:, it]
            if ci_model.do_sedim:
                budget_ice_sedim = ci_model.ds["budget_ice_sedim"].values[:, it]

        t_step_mix_mask = ci_model.ds["mixing_mask"].values[:, it - 1]  # mixed parts of profile = True

        for key in ci_model.aer.keys():
            if np.logical_or(ci_model.aer[key].is_INAS, ci_model.use_ABIFM):
                n_aer_prev = ci_model.aer[key].ds["n_aer"].values[:, it - 1, :]  # ptr: aerosol conc in prev. step.
                n_aer_curr = ci_model.aer[key].ds["n_aer"].values[:, it, :]  # ptr: aerosol conc. in current step.
                n_aer_curr += n_aer_prev
                diam_dim_l = ci_model.aer[key].ds["diam"].size
                if ci_model.aer[key].is_INAS:
                    n_inp_prev = ci_model.aer[key].ds["inp_snap"].copy().values  # copy: INP conc in prev. step.
                    n_inp_curr = ci_model.aer[key].ds["inp_snap"].values  # ptr: INP conc. in current step.
            else:  # no aerosol diameter information under singular so n_aer_curr is a 1-D array (nz).
                n_aer_prev = ci_model.aer[key].ds["n_aer"].values[:, it - 1]  # ptr: aerosol conc in prev. step.
                n_aer_curr = ci_model.aer[key].ds["n_aer"].values[:, it]  # ptr: aerosol conc. in current step.
                n_aer_curr += n_aer_prev
                n_inp_prev = ci_model.aer[key].ds["inp"].values[:, it - 1, :]  # ptr: INP conc in prev. time step.
                n_inp_curr = ci_model.aer[key].ds["inp"].values[:, it, :]  # ptr: INP conc. in current time step.
                n_inp_curr += n_inp_prev

            if ci_model.output_budgets:
                if np.logical_or(ci_model.aer[key].is_INAS, ci_model.use_ABIFM):
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

            # Activate aerosol
            t_process = time()
            if ci_model.use_ABIFM:
                AA, JJ = np.meshgrid(ci_model.aer[key].ds["surf_area"].values,
                                     ci_model.aer[key].ds["Jhet"].values[:, it - 1])
                aer_act = np.minimum(n_aer_prev * JJ * AA * ci_model.delta_t, n_aer_prev)
                if ci_model.nuc_RH_thresh is not None:
                    aer_act = np.where(np.tile(np.expand_dims(in_cld_mask[:, it - 1], axis=1), (1, diam_dim_l)),
                                       aer_act, 0.)
            else:
                TTi, TTm = np.meshgrid(ci_model.aer[key].ds["T"].values,
                                       np.where(ci_model.ds["ql"].values[:, it - 1] >= ci_model.in_cld_q_thresh,
                                                ci_model.ds["T"].values[:, it - 1], np.nan))  # ignore noncld cells
                if ci_model.aer[key].is_INAS:
                    aer_act = np.minimum(np.where(np.tile(np.expand_dims(TTi >= TTm, axis=1), (1, diam_dim_l, 1)),
                                                  n_inp_prev, 0), n_inp_prev)
                else:
                    aer_act = np.minimum(np.where(TTi >= TTm, n_inp_prev, 0), n_inp_prev)
            if ci_model.aer[key].is_INAS:
                ci_model.ds["nuc_rate"].values[:, it] += aer_act.sum(axis=(1, 2)) / ci_model.delta_t  # nuc. rate
            else:
                ci_model.ds["nuc_rate"].values[:, it] += aer_act.sum(axis=1) / ci_model.delta_t  # nucleation rate
            if ci_model.use_ABIFM:
                n_aer_curr -= aer_act  # Subtract from aerosol reservoir.
                n_ice_curr += aer_act.sum(axis=1)  # Add from aerosol reservoir (*currently*, w/o aerosol memory)
                if ci_model.output_budgets:
                    budget_aer_act -= aer_act / ci_model.delta_t
            else:
                n_inp_curr -= aer_act  # Subtract from aerosol reservoir (INP subset).
                if ci_model.aer[key].is_INAS:
                    n_ice_curr += aer_act.sum(axis=(1, 2))  # Add from reservoir (*currently*, w/o aerosol memory)
                else:
                    n_ice_curr += aer_act.sum(axis=1)  # Add from reservoir (*currently*, w/o aerosol memory)
                if ci_model.output_budgets:
                    budget_aer_act -= aer_act.sum(axis=inp_sum_dim) / ci_model.delta_t
            run_stats["activation_aer"] += (time() - t_process)
            t_proc += time() - t_process

            # Cloud-top entrainment of aerosol and/or INP (depending on scheme).
            if ci_model.do_entrain:
                t_process = time()
                if np.logical_or(ci_model.aer[key].is_INAS, ci_model.use_ABIFM):  # aerosol entrainment
                    if np.logical_and(
                        ci_model.deplete_entrained,
                        cth_ind[it - 1] < ci_model.mod_nz):  # reservoir depletion (if not at domain top)
                        if np.logical_and(cth_ind[it - 1] != -9999, cth_ind[it - 1] + 1 < ci_model.mod_nz):
                            aer_ent = ci_model.ds["w_e_ent"].values[it - 1] / \
                                ent_delta_z[it - 1] * ci_model.delta_t * \
                                (n_aer_curr[cth_ind[it - 1] + 1, :] - n_aer_curr[ent_delta_n_ind[it - 1], :])
                            n_aer_curr[cth_ind[it - 1] + 1, :] -= aer_ent  # update aerosol conc. just above cth.
                            n_aer_curr[ent_target_ind[it - 1], :] += aer_ent
                    else:  # assuming inf. domain top reservoir (t=0 s) and that cld top is at domain top.
                        aer_ent = ci_model.ds["w_e_ent"].values[it - 1] / \
                            ent_delta_z[it - 1] * ci_model.delta_t * \
                            (ci_model.aer[key].ds["n_aer"].values[cth_ind[it - 1], 0, :] -
                             n_aer_curr[ent_delta_n_ind[it - 1], :])
                        n_aer_curr[ent_target_ind[it - 1], :] += aer_ent
                    if ci_model.output_budgets:
                        budget_aer_ent += aer_ent / ci_model.delta_t
                else:  # 1-D processing of n_aer_curr for singular.
                    if np.logical_and(
                        ci_model.deplete_entrained,
                        cth_ind[it - 1] < ci_model.mod_nz):  # reservoir depletion (if not at domain top)
                        if np.logical_and(cth_ind[it - 1] != -9999, cth_ind[it - 1] + 1 < ci_model.mod_nz):
                            aer_ent = ci_model.ds["w_e_ent"].values[it - 1] / \
                                ent_delta_z[it - 1] * ci_model.delta_t * \
                                (n_aer_curr[cth_ind[it - 1] + 1] - n_aer_curr[ent_delta_n_ind[it - 1]])
                            n_aer_curr[cth_ind[it - 1] + 1] -= aer_ent  # update aerosol conc. just above cth.
                            n_aer_curr[ent_target_ind[it - 1]] += aer_ent
                    else:  # assuming inf. domain top reservoir (t=0 s) and that cld top is at domain top.
                        aer_ent = ci_model.ds["w_e_ent"].values[it - 1] / \
                            ent_delta_z[it - 1] * ci_model.delta_t * \
                            (ci_model.aer[key].ds["n_aer"].values[cth_ind[it - 1], 0] -
                             n_aer_curr[ent_delta_n_ind[it - 1]])
                        n_aer_curr[ent_target_ind[it - 1]] += aer_ent
                    if ci_model.output_budgets:
                        budget_aer_ent[it] += aer_ent / ci_model.delta_t
                if not ci_model.use_ABIFM:  # INP entrainment
                    if np.logical_and(
                        ci_model.deplete_entrained,
                        cth_ind[it - 1] < ci_model.mod_nz):  # reservoir depletion (if not at domain top)
                        if np.logical_and(cth_ind[it - 1] != -9999, cth_ind[it - 1] + 1 < ci_model.mod_nz):
                            if ci_model.aer[key].is_INAS:  # additional dim (diam) for INAS
                                inp_ent = ci_model.ds["w_e_ent"].values[it - 1] / \
                                    ent_delta_z.values[cth_ind[it - 1]] * ci_model.delta_t * \
                                    (n_inp_curr[cth_ind[it - 1] + 1, :, :] -
                                     n_inp_curr[ent_delta_n_ind[it - 1], :, :])
                                n_inp_curr[ent_target_ind[it - 1], :, :] += inp_ent
                                n_inp_curr[cth_ind[it - 1] + 1, :, :] -= inp_ent  # update INP conc. just above cth
                                if ci_model.output_budgets:
                                    budget_aer_ent += inp_ent.sum(axis=inp_sum_dim-1) / ci_model.delta_t
                            else:
                                inp_ent = ci_model.ds["w_e_ent"].values[it - 1] / \
                                    ent_delta_z[it - 1] * ci_model.delta_t * \
                                    (n_inp_curr[cth_ind[it - 1] + 1, :] - n_inp_curr[ent_delta_n_ind[it - 1], :])
                                n_inp_curr[ent_target_ind[it - 1], :] += inp_ent
                                n_inp_curr[cth_ind[it - 1] + 1, :] -= inp_ent  # update INP conc. just above cth.
                                if ci_model.output_budgets:
                                    budget_aer_ent[it] += inp_ent.sum(axis=inp_sum_dim-1) / ci_model.delta_t
                    else:  # assuming inf. domain top reservoir (t=0 s) and that cld top is at domain top.
                        if ci_model.aer[key].is_INAS:  # additional dim (diam) for INAS
                            inp_ent = ci_model.ds["w_e_ent"].values[it - 1] / \
                                ent_delta_z[it - 1] * ci_model.delta_t * \
                                (ci_model.aer[key].ds["inp_init"].values[cth_ind[it - 1], :, :] -
                                 n_inp_curr[ent_delta_n_ind[it - 1], :, :])
                            n_inp_curr[ent_target_ind[it - 1], :, :] += inp_ent
                            if ci_model.output_budgets:
                                budget_aer_ent += inp_ent.sum(axis=inp_sum_dim-1) / ci_model.delta_t
                        else:
                            inp_ent = (ci_model.ds["w_e_ent"].values[it - 1] / \
                                ent_delta_z[it - 1] * ci_model.delta_t * \
                                (ci_model.aer[key].ds["inp"].values[cth_ind[it - 1], 0, :] -
                                 n_inp_curr[ent_delta_n_ind[it - 1], :]))
                            n_inp_curr[ent_target_ind[it - 1], :] += inp_ent
                            if ci_model.output_budgets:
                                budget_aer_ent[it] += inp_ent.sum(axis=inp_sum_dim-1) / ci_model.delta_t
                run_stats["entrainment_aer"] += (time() - t_process)
                t_proc += time() - t_process

            # Turbulent mixing of aerosol
            if ci_model.do_mix_aer:
                t_process = time()
                if np.any(t_step_mix_mask):  # checking that some mixing takes place
                    if np.logical_or(ci_model.aer[key].is_INAS, ci_model.use_ABIFM):  # aerosol mixing
                        if np.all(t_step_mix_mask):  # Faster processing for fully mixed domain
                            aer_fully_mixed = np.nanmean(n_aer_curr, axis=0)
                            aer_mixing = ci_model.delta_t / ci_model.ds["tau_mix"].values[it - 1] * \
                                (np.tile(np.expand_dims(aer_fully_mixed, axis=0), (ci_model.mod_nz, 1)) -
                                 n_aer_curr)
                            n_aer_curr += aer_mixing
                            if ci_model.output_budgets:
                                budget_aer_mix += aer_mixing / ci_model.delta_t

                        else:
                            aer_fully_mixed = np.nanmean(np.where(np.tile(np.expand_dims(t_step_mix_mask, axis=1),
                                                                          (1, diam_dim_l)),
                                                                  n_aer_curr, np.nan), axis=0)
                            aer_mixing = ci_model.delta_t / ci_model.ds["tau_mix"].values[it - 1] * \
                                (np.tile(np.expand_dims(aer_fully_mixed, axis=0), (np.sum(t_step_mix_mask), 1)) -
                                 n_aer_curr[t_step_mix_mask, :])
                            n_aer_curr[t_step_mix_mask, :] += aer_mixing
                            if ci_model.output_budgets:
                                budget_aer_mix[t_step_mix_mask, :] += aer_mixing / ci_model.delta_t
                    else:
                        if np.all(t_step_mix_mask):  # Faster processing for fully mixed domain
                            aer_fully_mixed = np.nanmean(n_aer_curr)
                            aer_mixing = ci_model.delta_t / ci_model.ds["tau_mix"].values[it - 1] * \
                                (aer_fully_mixed - n_aer_curr)
                            n_aer_curr += aer_mixing
                            if ci_model.output_budgets:
                                budget_aer_mix += aer_mixing / ci_model.delta_t
                        else:
                            aer_fully_mixed = np.nanmean(np.where(t_step_mix_mask, n_aer_curr, np.nan))
                            aer_mixing = ci_model.delta_t / ci_model.ds["tau_mix"].values[it - 1] * \
                                (aer_fully_mixed - n_aer_curr[t_step_mix_mask])
                            n_aer_curr[t_step_mix_mask] += aer_mixing
                            if ci_model.output_budgets:
                                budget_aer_mix[t_step_mix_mask] += aer_mixing / ci_model.delta_t
                if not ci_model.use_ABIFM:  # INP mixing
                    if np.any(t_step_mix_mask):  # checking that some mixing takes place.
                        if np.all(t_step_mix_mask):  # Faster processing for fully mixed domain
                            inp_fully_mixed = np.nanmean(n_inp_curr, axis=0)
                            if ci_model.aer[key].is_INAS:  # additional dim (diam) for INAS
                                inp_mixing = ci_model.delta_t / ci_model.ds["tau_mix"].values[it - 1] * \
                                    (np.tile(np.expand_dims(inp_fully_mixed, axis=0), (ci_model.mod_nz, 1, 1)) -
                                     n_inp_curr)
                            else:
                                inp_mixing = ci_model.delta_t / ci_model.ds["tau_mix"].values[it - 1] * \
                                    (np.tile(np.expand_dims(inp_fully_mixed, axis=0), (ci_model.mod_nz, 1)) -
                                     n_inp_curr)
                            n_inp_curr += inp_mixing
                            if ci_model.output_budgets:
                                budget_aer_mix += inp_mixing.sum(axis=inp_sum_dim) / ci_model.delta_t
                        else:
                            if ci_model.aer[key].is_INAS:  # additional dim (diam) for INAS
                                inp_fully_mixed = np.nanmean(np.where(np.tile(np.expand_dims(t_step_mix_mask,
                                                                                             axis=(1, 2)),
                                                                              (1, diam_dim_l,
                                                                               ci_model.aer[key].ds["T"].size)),
                                                                      n_inp_curr, np.nan), axis=0)
                                inp_mixing = ci_model.delta_t / ci_model.ds["tau_mix"].values[it - 1] * \
                                    (np.tile(np.expand_dims(inp_fully_mixed, axis=0),
                                             (np.sum(t_step_mix_mask), 1, 1)) - n_inp_curr[t_step_mix_mask, :, :])
                                n_inp_curr[t_step_mix_mask, :, :] += inp_mixing
                                if ci_model.output_budgets:
                                    budget_aer_mix[t_step_mix_mask, :] += \
                                        inp_mixing.sum(axis=inp_sum_dim) / ci_model.delta_t
                            else:
                                inp_fully_mixed = np.nanmean(np.where(np.tile(np.expand_dims(t_step_mix_mask,
                                                                                             axis=1),
                                                                              (1, ci_model.aer[key].ds["T"].size)),
                                                                      n_inp_curr, np.nan), axis=0)
                                inp_mixing = ci_model.delta_t / ci_model.ds["tau_mix"].values[it - 1] * \
                                    (np.tile(np.expand_dims(inp_fully_mixed, axis=0),
                                             (np.sum(t_step_mix_mask), 1)) - n_inp_curr[t_step_mix_mask, :])
                                n_inp_curr[t_step_mix_mask, :] += inp_mixing
                                if ci_model.output_budgets:
                                    budget_aer_mix[t_step_mix_mask] += \
                                        inp_mixing.sum(axis=inp_sum_dim) / ci_model.delta_t
                run_stats["mixing_aer"] += (time() - t_process)
                t_proc += time() - t_process

            # Place resolved aerosol and/or INP
            if ci_model.use_ABIFM:
                ci_model.aer[key].ds["n_aer"][:, it, :].values = n_aer_curr
            elif ci_model.aer[key].is_INAS:
                ci_model.aer[key].ds["n_aer"][:, it, :].values = n_aer_curr
                ci_model.aer[key].ds["inp_snap"].values = n_inp_curr
                ci_model.aer[key].ds["inp_tot"][:, it, :].values += np.sum(n_inp_curr, axis=2)
            else:
                ci_model.aer[key].ds["inp"].values[:, it, :] = n_inp_curr
                ci_model.aer[key].ds["n_aer"][:, it].values = n_aer_curr

        # Sedimentation of ice (after aerosol were activated).
        if ci_model.do_sedim:
            t_process = time()
            if ci_model.ds["v_f_ice"].ndim == 2:
                ice_sedim_out = n_ice_curr * ci_model.ds["v_f_ice"].values[:, it - 1] * ci_model.delta_t / \
                    ci_model.ds["delta_z"].values
            else:
                ice_sedim_out = n_ice_curr * ci_model.ds["v_f_ice"].values[it - 1] * ci_model.delta_t / \
                    ci_model.ds["delta_z"].values
            ice_sedim_in = np.zeros(ci_model.mod_nz)
            ice_sedim_in[:-1] += ice_sedim_out[1:]
            n_ice_curr += ice_sedim_in
            n_ice_curr -= ice_sedim_out
            if ci_model.output_budgets:
                budget_ice_sedim += (ice_sedim_in - ice_sedim_out) / ci_model.delta_t
            t_proc += time() - t_process
            run_stats["sedimentation_ice"] += (time() - t_process)

        # Turbulent mixing of ice
        if ci_model.do_mix_ice:
            t_process = time()
            if np.any(t_step_mix_mask):  # checking that some mixing takes place.
                if np.all(t_step_mix_mask):  # Faster processing for fully mixed domain
                    ice_fully_mixed = np.nanmean(n_ice_curr, axis=0)
                    ice_mixing = ci_model.delta_t / ci_model.ds["tau_mix"].values[it - 1] * \
                        (ice_fully_mixed - n_ice_curr)
                    n_ice_curr += ice_mixing
                    if ci_model.output_budgets:
                        budget_ice_mix += ice_mixing / ci_model.delta_t
                else:
                    ice_fully_mixed = np.nanmean(np.where(t_step_mix_mask, n_ice_curr, np.nan), axis=0)
                    ice_mixing = ci_model.delta_t / ci_model.ds["tau_mix"].values[it - 1] * \
                        (ice_fully_mixed - n_ice_curr[t_step_mix_mask])
                    n_ice_curr[t_step_mix_mask] += ice_mixing
                    if ci_model.output_budgets:
                        budget_ice_mix[t_step_mix_mask] += ice_mixing / ci_model.delta_t
            t_proc += time() - t_process
            run_stats["mixing_ice"] += (time() - t_process)

        # Place resolved ice
        ci_model.ds["Ni_nuc"][:, it].values = n_ice_curr

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

        if ci_model.aer[key].is_INAS:
            ci_model.aer[key].ds["n_aer"].values += ci_model.aer[key].ds["inp_tot"].values
            sum_dims = (ci_model.height_dim, ci_model.diam_dim)  # for aer decay calculations if output_aer_decay
        elif np.logical_and(not ci_model.use_ABIFM, not ci_model.aer[key].is_INAS): 
            ci_model.aer[key].ds["n_aer"].values += ci_model.aer[key].ds["inp"].copy().sum("T").values
            sum_dims = (ci_model.height_dim)  # for aer decay calculations if output_aer_decay
        else:
            sum_dims = (ci_model.height_dim, ci_model.diam_dim)  # for aer decay calculations if output_aer_decay

        # Calculate decay statistics
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
                                     prepend=np.nan) / ci_model.delta_t * (-1),
                             dims=(ci_model.time_dim),
                             attrs={"units": "$m^{-2} s^{-1}$",
                                    "long_name": "Total PBL aerosol decay rate (multiplied by -1)"})

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
