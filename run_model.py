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
        1. INP activation since previous time step.
        2. Cloud-top entrainment of INP using entrainment rate from the previous time step.
        3. Turbulent mixing of INP using mixing depth and time scales from the previous time step (independent from
           the next steps).
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
    run_stats = {"activation_inp": 0, "entrainment_inp": 0, "mixing_inp": 0, "sedimentation_ice": 0,
                 "mixing_ice": 0, "data_allocation": 0}
    Complete_counter = 0
    print_delta_percent = 10.  # report model run progress every certain percentage of time steps.

    # find indices of cloud-top height
    if ci_model.entrain_from_cth:
        cth_ind = np.argmin(np.abs(np.tile(np.expand_dims(ci_model.ds["lowest_cth"].values, axis=0),
                            (ci_model.mod_nz, 1)) - np.tile(np.expand_dims(ci_model.ds["height"], axis=1),
                                                            (1, ci_model.mod_nt))), axis=0)
        cth_ind = np.where(cth_ind == 0, -9999, cth_ind)

    for it in range(1, ci_model.mod_nt):

        t_loop = time()  # counter for a single loop
        t_proc = 0

        if it / ci_model.mod_nt * 100 > Complete_counter + print_delta_percent:
            Complete_counter += print_delta_percent
            print("%.0f%% of model run completed. Elapsed time: %.2f s." % (Complete_counter, time() - Now))

        n_ice_prev = ci_model.ds["Ni_nuc"].values[:, it - 1]  # pointer for nucleated ice in prev. time step.
        n_ice_curr = ci_model.ds["Ni_nuc"].values[:, it]  # pointer for nucleated ice in current time step.
        n_ice_curr += n_ice_prev

        t_step_mix_mask = ci_model.ds["mixing_mask"].values[:, it - 1]  # mixed parts of profile = True

        for key in ci_model.inp.keys():
            n_inp_prev = ci_model.inp[key].ds["inp"].values[:, it - 1, :]  # pointer: INP conc in prev. time step.
            n_inp_curr = ci_model.inp[key].ds["inp"].values[:, it, :]  # pointer: INP conc. in current time step.
            n_inp_curr += n_inp_prev

            # Activate INP
            t_process = time()
            if ci_model.use_ABIFM:
                AA, JJ = np.meshgrid(ci_model.inp[key].ds["surf_area"].values,
                                     ci_model.inp[key].ds["Jhet"].values[:, it - 1])
                inp_act = np.minimum(n_inp_prev * JJ * AA * ci_model.delta_t, n_inp_prev)
            else:
                TTi, TTm = np.meshgrid(ci_model.inp[key].ds["T"].values,
                                       np.where(ci_model.ds["ql"].values[:, it - 1] >= ci_model.in_cld_q_thresh,
                                                ci_model.ds["T"].values[:, it - 1], np.nan))  # ignore noncld cells
                inp_act = np.minimum(np.where(TTi >= TTm, n_inp_prev, 0), n_inp_prev)
            ci_model.ds["nuc_rate"].values[:, it] += inp_act.sum(axis=1) / ci_model.delta_t  # nucleation rate
            n_inp_curr -= inp_act  # Subtract from inp reservoir.
            n_ice_curr += inp_act.sum(axis=1)  # Add from inp reservoir (*currently*, without INP memory)
            run_stats["activation_inp"] += (time() - t_process)
            t_proc += time() - t_process

            # Cloud-top entrainment of INP
            t_process = time()
            if ci_model.entrain_from_cth:  # using cloud top data (INP difference) for entrainment
                if np.logical_and(cth_ind[it - 1] != -9999, cth_ind[it - 1] + 1 < ci_model.mod_nz):
                    inp_ent = ci_model.ds["w_e_ent"].values[it - 1] / \
                        ci_model.ds["delta_z"].values[cth_ind[it - 1]] * ci_model.delta_t * \
                        (n_inp_curr[cth_ind[it - 1] + 1, :] - n_inp_curr[cth_ind[it - 1], :])
                    n_inp_curr[cth_ind[it - 1], :] += inp_ent
                    n_inp_curr[cth_ind[it - 1] + 1, :] -= inp_ent  # update INP conc. just above cloud top.
            else:  # assuming inf. domain top reservoir (t=0 s) and that cld top is at domain top.
                inp_ent = ci_model.ds["w_e_ent"].values[it - 1] / \
                    ci_model.ds["delta_z"].values[-1] * ci_model.delta_t * \
                    (ci_model.inp[key].ds["inp"].values[-1, 0, :] - n_inp_curr[-1, :])
                n_inp_curr[-1, :] += inp_ent
            run_stats["entrainment_inp"] += (time() - t_process)
            t_proc += time() - t_process

            # Turbulent mixing of INP
            t_process = time()
            if ci_model.use_ABIFM:
                PSDt = "diam"  # string for PSD dim (diameter in the case of singular)
            else:
                PSDt = "T"  # string for PSD dim (temperature in the case of singular)
            if np.any(t_step_mix_mask):  # checking that some mixing takes place.
                if np.all(t_step_mix_mask):  # Faster processing for fully mixed domain
                    inp_fully_mixed = np.nanmean(n_inp_curr, axis=0)
                    inp_mixing = ci_model.delta_t / ci_model.ds["tau_mix"].values[it - 1] * \
                        (np.tile(np.expand_dims(inp_fully_mixed, axis=0), (ci_model.mod_nz, 1)) - n_inp_curr)
                    n_inp_curr += inp_mixing
                else:
                    inp_fully_mixed = np.nanmean(np.where(np.tile(np.expand_dims(t_step_mix_mask, axis=1),
                                                                  (1, ci_model.inp[key].ds["inp"][PSDt].size)),
                                                          n_inp_curr, np.nan), axis=0)
                    inp_mixing = ci_model.delta_t / ci_model.ds["tau_mix"].values[it - 1] * \
                        (np.tile(np.expand_dims(inp_fully_mixed, axis=0), (np.sum(t_step_mix_mask), 1)) -
                         n_inp_curr[t_step_mix_mask, :])
                    n_inp_curr[t_step_mix_mask, :] += inp_mixing

                run_stats["mixing_inp"] += (time() - t_process)
                t_proc += time() - t_process

            # Place resolved INP
            ci_model.inp[key].ds["inp"][:, it, :].values = n_inp_curr

        # Sedimentation of ice (after INP were activated).
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
        t_proc += time() - t_process
        run_stats["sedimentation_ice"] += (time() - t_process)

        # Turbulent mixing of ice
        t_process = time()
        if np.any(t_step_mix_mask):  # checking that some mixing takes place.
            if np.all(t_step_mix_mask):  # Faster processing for fully mixed domain
                ice_fully_mixed = np.nanmean(n_ice_curr, axis=0)
                ice_mixing = ci_model.delta_t / ci_model.ds["tau_mix"].values[it - 1] * \
                    (ice_fully_mixed - n_ice_curr)
                n_ice_curr += ice_mixing
            else:
                ice_fully_mixed = np.nanmean(np.where(t_step_mix_mask, n_ice_curr, np.nan), axis=0)
                ice_mixing = ci_model.delta_t / ci_model.ds["tau_mix"].values[it - 1] * \
                    (ice_fully_mixed - n_ice_curr[t_step_mix_mask])
                n_ice_curr[t_step_mix_mask] += ice_mixing
        t_proc += time() - t_process
        run_stats["mixing_ice"] += (time() - t_process)

        # Place resolved ice
        ci_model.ds["Ni_nuc"][:, it].values = n_ice_curr

        run_stats["data_allocation"] += (time() - t_loop - t_proc)

    # Print model run summary
    runtime_tot = time() - Now
    print("\nModel run finished! Total run time = %f s\nModel run time stats:" % runtime_tot)
    for key in run_stats.keys():
        print("Process: %s: %.2f s (%.2f%% of of total time)" %
              (key, run_stats[key], run_stats[key] / runtime_tot * 100.))
