"""
This module provides multiple methods to plot model output.
"""
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors


def generate_figure(subplot_shape=(1,), figsize=(15, 10), **kwargs):
    """
    Generate a figure window - effectively the same as calling the 'plt.subplots' method, only with
    some default parameter values.

    Parameters
    ----------
    subplot_shape: 2-element tuple
        Determine the number of panel rows and columns, respectively.
    figsize: 2-element tuple
        Determine the figure's width and height, respectively.

    Returns
    -------
    fig: Matplotlib figure handle
    ax: Matplotlib axes handle
    """
    fig, ax = plt.subplots(*subplot_shape, figsize=figsize, **kwargs)
    return fig, ax


def curtain(ci_model, which_inp=None, field_to_plot="", x="time", y="height", inp_z=None, dim_treat="sum",
            cmap="cubehelix", vmin="auto", vmax="auto", ax=None, colorbar=True, cbar_label=None,
            xscale=None, yscale=None, log_plot=False, title=None, grid=False, xlabel=None, ylabel=None,
            tight_layout=True, font_size=None, xtick=None, xticklabel=None, ytick=None, yticklabel=None,
            xlim=None, ylim=None, **kwargs):
    """
    Generate a curtain plot of an INP population or another model field.

    Parameters
    ----------
    ci_model: ci_model class
        Containing model run output.
    which_inp: list, str, or None
        Name of INP population to plot. If field_to_plot is "Jhet" then plotting the INP population Jhet.
        If a list, then adding all inp populations together after checking that the "diam" (ABIFM) or "T"
        (singular) arrays have the same length and values.
        If None, then plot a field from the ci_model.ds xr.Dataset.
    field_to_plot: str
        Name of field to plot. If "Jhet" then remember to provide 'which_inp'.
    x: str
        coordinate to place on the x-axis - choose between "time", "height", "diam" (ABIFM), or "T" (singular).
    y: str
        coordinate to place on the y-axis - choose between "time", "height", "diam" (ABIFM), or "T" (singular).
    inp_z: float, int, 2-element tuple, or None
        Only for plotting of the INP field (ndim=3). Use a float to specify a 3rd dim coordinate value to use
        for plotting, int for 3rd coordinate index, tuple of floats to define a range of values (plotting a mean
        of that coordinate range), tuple of ints to define a range of indices (plotting a mean of that coordinate
        indices range), and None to plot mean over the full coordinate range.
    dim_treat: str
        Relevant if inp_z is a tuple or None. Use "mean", "sum", or "sd" for mean, sum, or standard deviation,
        respectively.
    cmap: str or matplotlib.cm.cmap
        colormap to use to use
    vmin: str or float
        if "auto" then using 5th percentil. If float, defining minimum colormap value
    vmax: str or float
        if "auto" then using 99th percentil. If float, defining maximum colormap value
    ax: matplotlib axis handle
        axes to plot on. If None, then creating a new figure.
    colorbar: bool
        Create a colorbar if true. Handle is returned in that case.
    cbar_label: str
        colorbar lable. Using the field name if None
    xscale: str or None
        "linear" or "log" scale for x-axis. If None, using "linear" for "time", "height", or "T" (singular)
        and "log" for "diam" (ABIFM).
    yscale: str or None
        "linear" or "log" scale for y-axis. If None, using "linear" for "time", "height", or "T" (singular)
        and "log" for "diam" (ABIFM).
    log_plot: bool
        scale for the c-axis (color-scale). Choose between linear (False) or log-scale (True).
    title: str or None
        panel (subplot) title if str
    grid: bool
        If True, add grid.
    xlabel: str or None
        set xlabel with str. Use field name and units if None
    ylabel: str or None
        set ylabel with str. Use field name and units if None
    tight_layout: bool
        If True, using a tight layout (no text overlap or text cut at figure edges) - typically, the
        desired behavior).
    xtick: list, np.ndarray, or None
        if provided, then used as x-axis ticks.
    xticklabel: list, np.ndarray, or None
        if provided, then used as x-axis tick labels.
    ytick: list, np.ndarray, or None
        if provided, then used as y-axis ticks
    yticklabel: list, np.ndarray, or None
        if provided, then used as y-axis tick labels.
    xlim: 2-elemnt tuple (or list) or None
        xlim range
    ylim: 2-elemnt tuple (or list) or None
        ylim range

    Returns
    -------
    ax: Matplotlib axes handle
    cb: Matplotlib colorbar handle (if 'colorbar' is True).
    """
    if which_inp is None:  # plot a field from ci_model.ds
        if field_to_plot in ci_model.ds.keys():
            plot_data = ci_model.ds[field_to_plot].copy()
            xf, yf = ci_model.ds[x], ci_model.ds[y]
        else:
            raise KeyError("Could not find the field: '%s' in ci_model.ds. Check for typos, etc." % field_to_plot)
    elif isinstance(which_inp, (list, str)):
        if isinstance(which_inp, str):
            if np.logical_and(which_inp in ci_model.inp.keys(), field_to_plot == "Jhet"):
                plot_data = ci_model.inp[which_inp].ds[field_to_plot].copy()
                xf, yf = ci_model.ds[x], ci_model.ds[y]
            else:
                which_inp = [which_inp]
        if np.logical_and(np.all([x in ci_model.inp.keys() for x in which_inp]), field_to_plot != "Jhet"):
            for ii in range(len(which_inp)):
                if ii == 0:
                    plot_data = ci_model.inp[which_inp[ii]].ds["inp"].copy()  # plot INP field
                    if ci_model.use_ABIFM:
                        inp_dim = "diam"
                    else:
                        inp_dim = "T"
                else:
                    interp_diams = False
                    if plot_data[inp_dim].size == ci_model.inp[which_inp[ii]].ds[inp_dim].size:
                        if np.all(plot_data[inp_dim].values == ci_model.inp[which_inp[ii]].ds[inp_dim].values):
                            plot_data += ci_model.inp[which_inp[ii]].ds["inp"]
                        else:
                            interp_diams = True
                            raise RuntimeError("different aerosol %s dim - %s arrays must have the same length \
                                               and values (interpolation will be added updated in future \
                                               updates)" % (inp_dim, inp_dim))
                    else:
                        interp_diams = True
                        raise RuntimeError("different aerosol %s dim - %s arrays must have the same length \
                                           and values (interpolation will be added updated in future updates)" \
                                           % (inp_dim, inp_dim))
            xf, yf = plot_data[x], plot_data[y]
            if ci_model.use_ABIFM:
                possible_fields = {"height", "time", "diam"}
            else:
                possible_fields = {"height", "time", "T"}
            [possible_fields.remove(fn) for fn in [x, y] if fn in possible_fields]
            if len(possible_fields) > 1:
                raise RuntimeError("something is not right - too many optional fields \
                                   (check 'x' and 'y' string values)")
            z = possible_fields.pop()
            plot_data = process_dim(plot_data, z, inp_z, dim_treat)
        elif field_to_plot != "Jhet":
            raise KeyError("Could not find one or more of the requested aerosl population names: \
                           '%s' in ci_model.inp. Check for typos, etc." % which_inp)

    # arrange plot dims
    if x == plot_data.dims[0]:
        plot_data = plot_data.transpose()

    if vmin == "auto":
        vmin = np.percentile(plot_data, 1)
    if vmax == "auto":
        vmax = np.percentile(plot_data, 99)

    if xlabel is None:
        xlabel = "%s [%s]" % (x, plot_data[x].attrs["units"])
    if ylabel is None:
        ylabel = "%s [%s]" % (y, plot_data[y].attrs["units"])

    if np.logical_and(xscale is None, x == "diam"):
        xscale = "log"
    if np.logical_and(yscale is None, y == "diam"):
        yscale = "log"

    if log_plot is True:
        mesh = ax.pcolormesh(xf, yf, plot_data, norm=colors.LogNorm(vmin=vmin, vmax=vmax),
                             cmap=cmap, **kwargs)
    else:
        mesh = ax.pcolormesh(xf, yf, plot_data, cmap=cmap,
                             vmin=vmin, vmax=vmax, **kwargs)

    fine_tuning(ax, title, xscale, yscale, grid, xlabel, ylabel, tight_layout, font_size, xtick, xticklabel,
                ytick, yticklabel, xlim, ylim)

    if colorbar:
        cb = plt.colorbar(mesh, ax=ax)
        if cbar_label is None:
            cb.set_label("%s" % plot_data.attrs["units"])
        else:
            cb.set_label(cbar_label)
        if font_size is not None:
            cb.ax.tick_params(labelsize=font_size)
            cb.ax.set_ylabel(cb.ax.get_yaxis().label.get_text(), fontsize=font_size)
        return ax, cb

    return ax


def tseries(ci_model, which_inp=None, field_to_plot="", inp_z=None, dim_treat="sum",
            Height=None, Height_dim_treat="mean", ax=None,
            yscale=None, title=None, grid=False, xlabel=None, ylabel=None, tight_layout=True,
            font_size=16, xtick=None, xticklabel=None, ytick=None, yticklabel=None, legend=None,
            xlim=None, ylim=None, **kwargs):
    """
    Generates INP or other model output field's time series

    Parameters
    ----------
    ci_model: ci_model class
        Containing model run output.
    field_to_plot: str
        Name of field to plot. If "Jhet" then remember to provide 'which_inp'.
    which_inp: str or None
        Name of INP population to plot. If field_to_plot is "Jhet" then plotting the INP population Jhet.
        If None, then plot a field from the ci_model.ds xr.Dataset.
    field_to_plot: str
        Name of field to plot. If "Jhet" then remember to provide 'which_inp'.
    inp_z: float, int, 2-element tuple, or None
        Only for plotting of the INP field (ndim=3). Use a float to specify a 3rd dim coordinate value to use
        for plotting, int for 3rd coordinate index, tuple of floats to define a range of values (plotting a mean
        of that coordinate range), tuple of ints to define a range of indices (plotting a mean of that coordinate
        indices range), and None to plot mean over the full coordinate range.
    dim_treat: str
        Relevant if inp_z is a tuple or None. Use "mean", "sum", or "sd" for mean, sum, or standard deviation,
        respectively.
    Height: float, int, 2-element tuple, list, or None
        Height elements, range, or values to treat from the dim. Options:
        1. float to specify a dim coordinate value to use for plotting.
        2. int for coordinate index.
        3. tuple of floats to define a range of values.
        4. tuple of ints to define a range of indices.
        5. list or np.darray of floats to define a specific values.
        6. list or np.darray of ints to define a specific indices.
        7. None to take the full coordinate range.
    Height_dim_treat: str
        How to treat the height dimension. Use "mean", "sum", "sd", None for mean, sum, standard deviation,
        non-treatment (keep dim),
        respectively.
    ax: matplotlib axis handle
        axes to plot on. If None, then creating a new figure.
    yscale: str or None
        "linear" or "log" scale for y-axis. If None, using "linear" for "time", "height", or "T" (singular)
        and "log" for "diam" (ABIFM).
    title: str or None
        panel (subplot) title if str
    grid: bool
        If True, add grid.
    xlabel: str or None
        set xlabel with str. Use field name and units if None
    ylabel: str or None
        set ylabel with str. Use field name and units if None
    tight_layout: bool
        If True, using a tight layout (no text overlap or text cut at figure edges) - typically, the
        desired behavior).
    xtick: list, np.ndarray, or None
        if provided, then used as x-axis ticks.
    xticklabel: list, np.ndarray, or None
        if provided, then used as x-axis tick labels.
    ytick: list, np.ndarray, or None
        if provided, then used as y-axis ticks
    yticklabel: list, np.ndarray, or None
        if provided, then used as y-axis tick labels.
    xlim: 2-elemnt tuple (or list) or None
        xlim range
    ylim: 2-elemnt tuple (or list) or None
        ylim range
    legend: bool or None
        if None, placing legend if processed plot_data has ndim = 2.

    Returns
    -------
    ax: Matplotlib axes handle
    """
    if which_inp is None:  # plot a field from ci_model.ds
        if field_to_plot in ci_model.ds.keys():
            plot_data = ci_model.ds[field_to_plot].copy()
            label = field_to_plot
        else:
            raise KeyError("Could not find the field: '%s' in ci_model.ds. Check for typos, etc." % field_to_plot)
    elif isinstance(which_inp, (list, str)):
        if isinstance(which_inp, str):
            if np.logical_and(which_inp in ci_model.inp.keys(), field_to_plot == "Jhet"):
                plot_data = ci_model.inp[which_inp].ds[field_to_plot].copy()
                label = "%s %s" % (which_inp, "Jhet")
            else:
                which_inp = [which_inp]
        if np.logical_and(np.all([x in ci_model.inp.keys() for x in which_inp]), field_to_plot != "Jhet"):
            for ii in range(len(which_inp)):
                if ii == 0:
                    plot_data = ci_model.inp[which_inp[ii]].ds["inp"].copy()  # plot INP field
                    label = "%s %s" % (which_inp[ii], "INP")
                    if ci_model.use_ABIFM:
                        inp_dim = "diam"
                    else:
                        inp_dim = "T"
                else:
                    interp_diams = False
                    if plot_data[inp_dim].size == ci_model.inp[which_inp[ii]].ds[inp_dim].size:
                        if np.all(plot_data[inp_dim].values == ci_model.inp[which_inp[ii]].ds[inp_dim].values):
                            plot_data += ci_model.inp[which_inp[ii]].ds["inp"]
                        else:
                            interp_diams = True
                            raise RuntimeError("different aerosol %s dim - %s arrays must have the same length \
                                               and values (interpolation will be added updated in future \
                                               updates)" % (inp_dim, inp_dim))
                    else:
                        interp_diams = True
                        raise RuntimeError("different aerosol %s dim - %s arrays must have the same length \
                                           and values (interpolation will be added updated in future updates)" \
                                           % (inp_dim, inp_dim))
            plot_data = process_dim(plot_data, inp_dim, inp_z, dim_treat)
        elif field_to_plot != "Jhet":
            raise KeyError("Could not find one or more of the requested aerosl population names: \
                           '%s' in ci_model.inp. Check for typos, etc." % which_inp)

    # Select values or indices from the height dim and treat (mean, sum, as-is).
    if plot_data.ndim == 2:
        plot_data = process_dim(plot_data, "height", Height, Height_dim_treat)

    if xlabel is None:
        xlabel = "%s [%s]" % ("time", plot_data["time"].attrs["units"])
    if ylabel is None:
        ylabel = "%s [%s]" % (label, plot_data.attrs["units"])

    if plot_data.ndim == 3:
        raise RuntimeError("processed INP field still had 3 dimensions. Consider reducing by selecting \
                           a single values or indices or average/sum")
    elif plot_data.ndim == 2:
        dim_2nd = [x for x in plot_data.dims if x != "time"][0]  # dimension to loop over (height unless INP).
        for ii in range(plot_data[dim_2nd].size):
            if "units" in plot_data[dim_2nd].attrs:
                label_p = label + " (%s = %.1f %s)" % (dim_2nd, plot_data[dim_2nd][ii],
                                                       plot_data[dim_2nd].attrs["units"])
            else:
                label_p = label + " (%s = %.1f)" % (dim_2nd, plot_data[dim_2nd][ii])
            ax.plot(plot_data["time"], plot_data.isel({dim_2nd: ii}), label=label_p, **kwargs)
        if legend is None:
            legend = True
    else:
        ax.plot(plot_data["time"], plot_data, label=label, **kwargs)

    ax = fine_tuning(ax, title, None, yscale, grid, xlabel, ylabel, tight_layout, font_size, xtick, xticklabel,
                     ytick, yticklabel, xlim, ylim)

    if legend is True:
        ax.legend()

    return ax


def profile(ci_model, which_inp=None, field_to_plot="", inp_z=None, dim_treat="sum",
            Time=None, Time_dim_treat="mean", ax=None,
            xscale=None, title=None, grid=False, xlabel=None, ylabel=None, tight_layout=True,
            font_size=16, xtick=None, xticklabel=None, ytick=None, yticklabel=None, legend=None,
            xlim=None, ylim=None, **kwargs):

    """
    Generates INP or other model output field's profile

    Parameters
    ----------
    ci_model: ci_model class
        Containing model run output.
    field_to_plot: str
        Name of field to plot. If "Jhet" then remember to provide 'which_inp'.
    which_inp: str or None
        Name of INP population to plot. If field_to_plot is "Jhet" then plotting the INP population Jhet.
        If None, then plot a field from the ci_model.ds xr.Dataset.
    field_to_plot: str
        Name of field to plot. If "Jhet" then remember to provide 'which_inp'.
    inp_z: float, int, 2-element tuple, or None
        Only for plotting of the INP field (ndim=3). Use a float to specify a 3rd dim coordinate value to use
        for plotting, int for 3rd coordinate index, tuple of floats to define a range of values (plotting a mean
        of that coordinate range), tuple of ints to define a range of indices (plotting a mean of that coordinate
        indices range), and None to plot mean over the full coordinate range.
    dim_treat: str
        Relevant if inp_z is a tuple or None. Use "mean", "sum", or "sd" for mean, sum, or standard deviation,
        respectively.
    Time: float, int, 2-element tuple, list, or None
        Time elements, range, or values to treat from the dim. Options:
        1. float to specify a dim coordinate value to use for plotting.
        2. int for coordinate index.
        3. tuple of floats to define a range of values.
        4. tuple of ints to define a range of indices.
        5. list or np.darray of floats to define a specific values.
        6. list or np.darray of ints to define a specific indices.
        7. None to take the full coordinate range.
    Time_dim_treat: str
        How to treat the time dimension. Use "mean", "sum", "sd", None for mean, sum, standard deviation,
        non-treatment (keep dim),
        respectively.
    ax: matplotlib axis handle
        axes to plot on. If None, then creating a new figure.
    xscale: str or None
        "linear" or "log" scale for x-axis. If None, using "linear" for "time", "height", or "T" (singular)
        and "log" for "diam" (ABIFM).
    title: str or None
        panel (subplot) title if str
    grid: bool
        If True, add grid.
    xlabel: str or None
        set xlabel with str. Use field name and units if None
    ylabel: str or None
        set ylabel with str. Use field name and units if None
    tight_layout: bool
        If True, using a tight layout (no text overlap or text cut at figure edges) - typically, the
        desired behavior).
    xtick: list, np.ndarray, or None
        if provided, then used as x-axis ticks.
    xticklabel: list, np.ndarray, or None
        if provided, then used as x-axis tick labels.
    ytick: list, np.ndarray, or None
        if provided, then used as y-axis ticks
    yticklabel: list, np.ndarray, or None
        if provided, then used as y-axis tick labels.
    xlim: 2-elemnt tuple (or list) or None
        xlim range
    ylim: 2-elemnt tuple (or list) or None
        ylim range
    legend: bool or None
        if None, placing legend if processed plot_data has ndim = 2.

    Returns
    -------
    ax: Matplotlib axes handle
    """
    if which_inp is None:  # plot a field from ci_model.ds
        if field_to_plot in ci_model.ds.keys():
            plot_data = ci_model.ds[field_to_plot].copy()
            label = field_to_plot
        else:
            raise KeyError("Could not find the field: '%s' in ci_model.ds. Check for typos, etc." % field_to_plot)
    elif isinstance(which_inp, (list, str)):
        if isinstance(which_inp, str):
            if np.logical_and(which_inp in ci_model.inp.keys(), field_to_plot == "Jhet"):
                plot_data = ci_model.inp[which_inp].ds[field_to_plot].copy()
                label = "%s %s" % (which_inp, "Jhet")
            else:
                which_inp = [which_inp]
        if np.logical_and(np.all([x in ci_model.inp.keys() for x in which_inp]), field_to_plot != "Jhet"):
            for ii in range(len(which_inp)):
                if ii == 0:
                    plot_data = ci_model.inp[which_inp[ii]].ds["inp"].copy()  # plot INP field
                    label = "%s %s" % (which_inp[ii], "INP")
                    if ci_model.use_ABIFM:
                        inp_dim = "diam"
                    else:
                        inp_dim = "T"
                else:
                    interp_diams = False
                    if plot_data[inp_dim].size == ci_model.inp[which_inp[ii]].ds[inp_dim].size:
                        if np.all(plot_data[inp_dim].values == ci_model.inp[which_inp[ii]].ds[inp_dim].values):
                            plot_data += ci_model.inp[which_inp[ii]].ds["inp"]
                        else:
                            interp_diams = True
                            raise RuntimeError("different aerosol %s dim - %s arrays must have the same length \
                                               and values (interpolation will be added updated in future \
                                               updates)" % (inp_dim, inp_dim))
                    else:
                        interp_diams = True
                        raise RuntimeError("different aerosol %s dim - %s arrays must have the same length \
                                           and values (interpolation will be added updated in future updates)" \
                                           % (inp_dim, inp_dim))
            plot_data = process_dim(plot_data, inp_dim, inp_z, dim_treat)
        elif field_to_plot != "Jhet":
            raise KeyError("Could not find one or more of the requested aerosl population names: \
                           '%s' in ci_model.inp. Check for typos, etc." % which_inp)

    # Select values or indices from the time dim and treat (mean, sum, as-is).
    if plot_data.ndim == 2:
        plot_data = process_dim(plot_data, "time", Time, Time_dim_treat)

    if ylabel is None:
        ylabel = "%s [%s]" % ("height", plot_data["height"].attrs["units"])
    if xlabel is None:
        xlabel = "%s [%s]" % (label, plot_data.attrs["units"])

    if plot_data.ndim == 3:
        raise RuntimeError("processed INP field still had 3 dimensions. Consider reducing by selecting \
                           a single values or indices or average/sum")
    elif plot_data.ndim == 2:
        dim_2nd = [x for x in plot_data.dims if x != "height"][0]  # dimension to loop over (time unless INP).
        for ii in range(plot_data[dim_2nd].size):
            if "units" in plot_data[dim_2nd].attrs:
                label_p = label + " (%s = %.1f %s)" % (dim_2nd, plot_data[dim_2nd][ii],
                                                       plot_data[dim_2nd].attrs["units"])
            else:
                label_p = label + " (%s = %.1f)" % (dim_2nd, plot_data[dim_2nd][ii])
            ax.plot(plot_data.isel({dim_2nd: ii}), plot_data["height"], label=label_p, **kwargs)
        if legend is None:
            legend = True
    else:
        ax.plot(plot_data, plot_data["height"], label=label, **kwargs)

    ax = fine_tuning(ax, title, xscale, None, grid, xlabel, ylabel, tight_layout, font_size, xtick, xticklabel,
                     ytick, yticklabel, xlim, ylim)

    if legend is True:
        ax.legend()

    return ax



def PSD(ci_model, which_inp=None,
        Time=None, Time_dim_treat=None, Height=None, Height_dim_treat=None, ax=None,
        xscale=None, yscale=None, title=None, grid=False, xlabel=None, ylabel=None, tight_layout=True,
        font_size=16, xtick=None, xticklabel=None, ytick=None, yticklabel=None, legend=None,
        xlim=None, ylim=None, **kwargs):

    """
    Generates an aerosol population PSD plots

    Parameters
    ----------
    ci_model: ci_model class
        Containing model run output.
    which_inp: str or None
        Name of INP population to plot. If field_to_plot is "Jhet" then plotting the INP population Jhet.
        If None, then plot a field from the ci_model.ds xr.Dataset.
    Time: float, int, 2-element tuple, list, or None
        Time elements, range, or values to treat from the dim. Options:
        1. float to specify a dim coordinate value to use for plotting.
        2. int for coordinate index.
        3. tuple of floats to define a range of values.
        4. tuple of ints to define a range of indices.
        5. list or np.darray of floats to define a specific values.
        6. list or np.darray of ints to define a specific indices.
        7. None to take the full coordinate range.
    Time_dim_treat: str
        How to treat the time dimension. Use "mean", "sum", "sd", None for mean, sum, standard deviation,
        non-treatment (keep dim), respectively.
    Height: float, int, 2-element tuple, list, or None
        Height elements, range, or values to treat from the dim. Options:
        1. float to specify a dim coordinate value to use for plotting.
        2. int for coordinate index.
        3. tuple of floats to define a range of values.
        4. tuple of ints to define a range of indices.
        5. list or np.darray of floats to define a specific values.
        6. list or np.darray of ints to define a specific indices.
        7. None to take the full coordinate range.
    Height_dim_treat: str
        How to treat the height dimension. Use "mean", "sum", "sd", None for mean, sum, standard deviation,
        non-treatment (keep dim), respectively.
    ax: matplotlib axis handle
        axes to plot on. If None, then creating a new figure.
    xscale: str or None
        "linear" or "log" scale for x-axis. If None, using "linear" for "time", "height", or "T" (singular)
        and "log" for "diam" (ABIFM).
    yscale: str or None
        "linear" or "log" scale for y-axis. If None, using "linear" for "time", "height", or "T" (singular)
        and "log" for "diam" (ABIFM).
    title: str or None
        panel (subplot) title if str
    grid: bool
        If True, add grid.
    xlabel: str or None
        set xlabel with str. Use field name and units if None
    ylabel: str or None
        set ylabel with str. Use field name and units if None
    tight_layout: bool
        If True, using a tight layout (no text overlap or text cut at figure edges) - typically, the
        desired behavior).
    xtick: list, np.ndarray, or None
        if provided, then used as x-axis ticks.
    xticklabel: list, np.ndarray, or None
        if provided, then used as x-axis tick labels.
    ytick: list, np.ndarray, or None
        if provided, then used as y-axis ticks
    yticklabel: list, np.ndarray, or None
        if provided, then used as y-axis tick labels.
    xlim: 2-elemnt tuple (or list) or None
        xlim range
    ylim: 2-elemnt tuple (or list) or None
        ylim range
    legend: bool or None
        if None, placing legend if processed plot_data has ndim >= 2.

    Returns
    -------
    ax: Matplotlib axes handle
    """
    if isinstance(which_inp, str):
        which_inp = [which_inp]
    if np.all([x in ci_model.inp.keys() for x in which_inp]):
        for ii in range(len(which_inp)):
            if ii == 0:
                plot_data = ci_model.inp[which_inp[ii]].ds["inp"].copy()  # plot INP field
                label = "%s %s" % (which_inp[ii], "PSD")
                if ci_model.use_ABIFM:
                    inp_dim = "diam"
                else:
                    inp_dim = "T"
            else:
                interp_diams = False
                if plot_data[inp_dim].size == ci_model.inp[which_inp[ii]].ds[inp_dim].size:
                    if np.all(plot_data[inp_dim].values == ci_model.inp[which_inp[ii]].ds[inp_dim].values):
                        plot_data += ci_model.inp[which_inp[ii]].ds["inp"]
                    else:
                        interp_diams = True
                        raise RuntimeError("different aerosol %s dim - %s arrays must have the same length \
                                           and values (interpolation will be added updated in future \
                                           updates)" % (inp_dim, inp_dim))
                else:
                    interp_diams = True
                    raise RuntimeError("different aerosol %s dim - %s arrays must have the same length \
                                       and values (interpolation will be added updated in future updates)" \
                                       % (inp_dim, inp_dim))
    else:
        raise KeyError("Could not find one or more of the requested aerosl population names: \
                       '%s' in ci_model.inp. Check for typos, etc." % which_inp)

    # Select values or indices from the time and height dims and treat (mean, sum, as-is).
    plot_data = process_dim(plot_data, "time", Time, Time_dim_treat)
    plot_data = process_dim(plot_data, "height", Height, Height_dim_treat)

    if xlabel is None:
        xlabel = "%s [%s]" % ("Diameter", plot_data["diam"].attrs["units"])
    if ylabel is None:
        ylabel = "%s [%s]" % (label, plot_data.attrs["units"])

    if plot_data.ndim == 3:
        plot_data = plot_data.stack(h_t=("height", "time"))
        heights = plot_data["height"].values
        times = plot_data["time"].values
    elif plot_data.ndim == 2:
        if "time" in plot_data.dims:
            times = plot_data["time"].values
            heights = None
            plot_data = plot_data.rename({"time": "h_t"})
        else:
            times = None
            heights = plot_data["height"].values
            plot_data = plot_data.rename({"height": "h_t"})
    else:
        heights = None
        times = None
        plot_data = plot_data.expand_dims("h_t")

    #return plot_data

    for ii in range(plot_data["h_t"].size):
        label_p = label
        if heights is not None:
            label_p = label_p + "; h = %.0f m" % (heights[ii])
        if times is not None:
            label_p = label_p + "; t = %.0f s" % (times[ii])

        ax.plot(plot_data[inp_dim], plot_data.isel({"h_t": ii}).values, label=label_p, **kwargs)
    if legend is None:
        legend = True

    ax = fine_tuning(ax, title, xscale, yscale, grid, xlabel, ylabel, tight_layout, font_size, xtick, xticklabel,
                     ytick, yticklabel, xlim, ylim)

    if legend is True:
        ax.legend()

    return ax


def fine_tuning(ax, title=None, xscale=None, yscale=None, grid=False, xlabel=None, ylabel=None,
                tight_layout=True, font_size=None, xtick=None, xticklabel=None, ytick=None,
                yticklabel=None, xlim=None, ylim=None):
    """
    Fine tune plot (labels, grids, etc.).

    Parameters
    ----------
    ax: matplotlib axis handle
        axes to plot on. If None, then creating a new figure.
    title: str or None
        panel (subplot) title if str
    xscale: str or None
        "linear" or "log" scale for x-axis. If None, using "linear" for "time", "height", or "T" (singular)
        and "log" for "diam" (ABIFM).
    yscale: str or None
        "linear" or "log" scale for y-axis. If None, using "linear" for "time", "height", or "T" (singular)
        and "log" for "diam" (ABIFM).
    log_plot: bool
        scale for the c-axis (color-scale). Choose between linear (False) or log-scale (True).
    grid: bool
        If True, add grid.
    xlabel: str or None
        set xlabel with str. Use field name and units if None
    ylabel: str or None
        set ylabel with str. Use field name and units if None
    tight_layout: bool
        If True, using a tight layout (no text overlap or text cut at figure edges) - typically, the
        desired behavior).
    font_size: float or None
        set font size in panel
    xtick: list, np.ndarray, or None
        if provided, then used as x-axis ticks.
    xticklabel: list, np.ndarray, or None
        if provided, then used as x-axis tick labels.
    ytick: list, np.ndarray, or None
        if provided, then used as y-axis ticks
    yticklabel: list, np.ndarray, or None
        if provided, then used as y-axis tick labels.
    xlim: 2-elemnt tuple (or list) or None
        xlim range
    ylim: 2-elemnt tuple (or list) or None
        ylim range
    """
    if title is not None:
        ax.set_title(title)

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if grid:
        ax.grid()

    if xscale is not None:
        ax.set_xscale(xscale)
    if yscale is not None:
        ax.set_yscale(yscale)

    if font_size is not None:
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(font_size)

    if xtick is not None:
        ax.set_xticks(xtick)
    if ytick is not None:
        ax.set_yticks(ytick)

    if xticklabel is not None:
        ax.set_xticklabels(xticklabel)
    if yticklabel is not None:
        ax.set_yticklabels(yticklabel)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    if tight_layout:
        plt.tight_layout()

    return ax


def process_dim(plot_data, dim_name, dim_vals_inds, dim_treat="sum"):
    """
    Process non-depicted dimension by cropping, slicing, averaging, summing, or calculating SD.

    Parameters
    ----------
    plot_data: xr.DataArray
        data array to be plotted.
    dim_name: str
        name of array coordinates to treat.
    dim_vals_inds: float, int, 2-element tuple, list, or None
        elements, range, or values to treat from the dim. Options:
        1. float to specify a dim coordinate value to use for plotting.
        2. int for coordinate index.
        3. tuple of floats to define a range of values.
        4. tuple of ints to define a range of indices.
        5. list or np.darray of floats to define a specific values.
        6. list or np.darray of ints to define a specific indices.
        7. None to take the full coordinate range.
    dim_treat: str
        Relevant if dim_vals_inds is a tuple, list, np.ndarray, or None.
        Use "mean", "sum", or "sd" for mean, sum, or standard deviation, respectively.

    Returns
    -------
    plot_data: xr.DataArray
        data array to be plotted.

    """
    dim_ind = np.argwhere([dim_name == x for x in plot_data.dims]).item()
    if dim_treat is None:
        treat_fun = lambda x: x  # Do nothing.
    elif dim_treat == "sum":
        treat_fun = lambda x: np.sum(x, axis=dim_ind)
    elif dim_treat == "mean":
        treat_fun = lambda x: np.mean(x, axis=dim_ind)
    elif dim_treat == "sd":
        treat_fun = lambda x: np.std(x, axis=dim_ind)
    else:
        raise RuntimeError("'dim_treat' should be one of 'sum', 'mean', 'sd', or None")

    units = ""
    if "units" in plot_data.attrs:
        units = plot_data.attrs["units"]
    if dim_vals_inds is None:
        plot_data = treat_fun(plot_data)
    elif isinstance(dim_vals_inds, float):
        plot_data = plot_data.sel({dim_name: dim_vals_inds})
    elif isinstance(dim_vals_inds, int):
        plot_data = plot_data.isel({dim_name: dim_vals_inds})
    elif isinstance(dim_vals_inds, tuple):
        if len(dim_vals_inds) != 2:
            raise RuntimeError("tuple (range) length must be 2")
        if isinstance(dim_vals_inds[0], float):  # check type of first index
            plot_data = treat_fun(plot_data.sel({dim_name: slice(dim_vals_inds[0], dim_vals_inds[1])}))
        else:
            plot_data = treat_fun(plot_data.isel({dim_name: slice(dim_vals_inds[0], dim_vals_inds[1])}))
    elif isinstance(dim_vals_inds, (list, np.ndarray)):
        if isinstance(dim_vals_inds[0], float):  # check type of first index
            plot_data = treat_fun(plot_data.sel({dim_name: dim_vals_inds}, method="nearest"))
        else:
            plot_data = treat_fun(plot_data.isel({dim_name: dim_vals_inds}))

    # restore units
    if units != "":
        plot_data.attrs["units"] = units

    return plot_data
