# -*- coding: utf-8 -*-
"""
Module for animating solutions in 1D.
Can also save them but requieres ffmpeg package
see save_animation method.
"""
import numpy as nm
from os.path import join as pjoin
from glob import glob

from matplotlib import animation
from matplotlib import pyplot as plt
from matplotlib import colors


from sfepy.discrete.fem.meshio import MeshioLibIO
from sfepy.discrete.fem.mesh import Mesh

# This would still use some refactoring so it is more flexible
__author__ = 'tomas_zitka'

ffmpeg_path = ''  # for saving animations

def head(l):
    """
    Maybe get head of the list.

    Parameters
    ----------
    l : indexable

    Returns
    -------
    head : first element in l or None is l is empty
    """
    if l:
        return l[0]
    else:
        return None


def animate1D_dgsol(Y, X, T, ax=None, fig=None, ylims=None, labs=None,
                    plott=None, delay=None):
    """Animates solution of 1D problem into current figure.
    Keep reference to returned animation object otherwise
    it is discarded

    Parameters
    ----------
    Y :
        solution, array |T| x |X| x n, where n is dimension of the solution
    X :
        space interval discetization
    T :
        time interval discretization
    ax :
        specify axes to plot to (Default value = None)
    fig :
        specifiy figure to plot to (Default value = None)
    ylims :
        limits for y axis, default are 10% offsets of Y extremes
    labs :
        labels to use for parts of the solution (Default value = None)
    plott :
        plot type - how to plot data: tested plot, step (Default value = None)
    delay :
         (Default value = None)

    Returns
    -------
    anim
        the animation object, keep it to see the animation, used for savig too
    """

    ax, fig, time_text = setup_axis(X, Y, ax, fig, ylims)

    if not isinstance(Y, nm.ndarray):
        Y = nm.stack(Y, axis=2)

    lines = setup_lines(ax, Y.shape, labs, plott)

    def animate(i):
        ax.legend()
        time_text.set_text("t= {0:3.2f} / {1:3.3}".format(T[i], T[-1]))
        # from sfepy.base.base import debug;
        # debug()
        if len(Y.shape) > 2:
            for ln, l in enumerate(lines):
                l.set_data(X, Y[i].swapaxes(0, 1)[ln])
            return tuple(lines) + (time_text,)
        # https://stackoverflow.com/questions/20624408/matplotlib-animating-multiple-lines-and-text
        else:
            lines.set_data(X, Y[i])
            return lines, time_text

    if delay is None:
        delay = int(nm.round(2000 * (T[-1] - T[0]) / len(T)))
    anim = animation.FuncAnimation(fig, animate, frames=len(T), interval=delay,
                                   blit=True, repeat=True, repeat_delay=250)

    return anim


def setup_axis(X, Y, ax=None, fig=None, ylims=None):
    """Setup axis, including timer for animation or snaps

    Parameters
    ----------
    X :
        space disctretization to get limits
    Y :
        solution to get limits
    ax :
        ax where to put everything, if None current axes are used (Default value = None)
    fig :
        fig where to put everything, if None current figure is used (Default value = None)
    ylims :
        custom ylims, if None y axis limits are calculated from Y (Default value = None)

    Returns
    -------
    ax

    fig

    time_text
        object to fill in text

    """
    if ax is None:
        fig = plt.gcf()
        ax = plt.gca()
    if ylims is None:
        lowery = nm.min(Y) - nm.min(Y) / 10
        uppery = nm.max(Y) + nm.max(Y) / 10
    else:
        lowery = ylims[0]
        uppery = ylims[1]
    ax.set_ylim(lowery, uppery)
    ax.set_xlim(X[0], X[-1])
    time_text = ax.text(X[0] + nm.sign(X[0]) * X[0] / 10,
                        uppery - uppery / 10,
                        'empty', fontsize=15)
    return ax, fig, time_text


def setup_lines(ax, Yshape, labs, plott):
    """Sets up artist for animation or solution snaps

    Parameters
    ----------
    ax :
        axes to use for artist
    Yshape : tuple
        shape of the solution array
    labs : list
        labels for the solution
    plott : str ("steps" or "plot")
        type of plot to use

    Returns
    -------

    lines
    """
    if plott is None:
        plott = ax.plot
    else:
        plott = ax.__getattribute__(plott)

    if len(Yshape) > 2:
        lines = [plott([], [], lw=2)[0] for foo in range(Yshape[2])]
        for i, l in enumerate(lines):
            if labs is None:
                l.set_label("q" + str(i + 1) + "(x, t)")
            else:
                l.set_label(labs[i])
    else:
        lines, = plott([], [], lw=2)
        if labs is None:
            lines.set_label("q(x, t)")
        else:
            lines.set_label(labs)
    return lines


def save_animation(anim, filename):
    """Saves animation as .mp4, requires ffmeg package

    Parameters
    ----------
    anim :
        animation object
    filename :
        name of the file, without the .mp4 ending
    """
    plt.rcParams['animation.ffmpeg_path'] = ffmpeg_path
    writer = animation.FFMpegWriter(fps=24)
    anim.save(filename + ".mp4", writer=writer)


def sol_frame(Y, X, T, t0=.5, ax=None, fig=None, ylims=None, labs=None, plott=None):
    """Creates snap of solution at specified time frame t0, basically gets one
    frame from animate1D_dgsol, but colors wont be the same :-(

    Parameters
    ----------
    Y :
        solution, array |T| x |X| x n, where n is dimension of the solution
    X :
        space interval discetization
    T :
        time interval discretization
    t0 :
        time to take snap at (Default value = .5)
    ax :
        specify axes to plot to (Default value = None)
    fig :
        specifiy figure to plot to (Default value = None)
    ylims :
        limits for y axis, default are 10% offsets of Y extremes
    labs :
        labels to use for parts of the solution (Default value = None)
    plott :
        plot type - how to plot data: tested plot, step (Default value = None)

    Returns
    -------
    fig
    """

    ax, fig, time_text = setup_axis(X, Y, ax, fig, ylims)

    if not isinstance(Y, nm.ndarray):
        Y = nm.stack(Y, axis=2)

    lines = setup_lines(ax, Y.shape, labs, plott)

    nt0 = nm.abs(T - t0).argmin()

    ax.legend()
    time_text.set_text("t= {0:3.2f} / {1:3.3}".format(T[nt0], T[-1]))
    if len(Y.shape) > 2:
        for ln, l in enumerate(lines):
            l.set_data(X, Y[nt0].swapaxes(0, 1)[ln])
    else:
        lines.set_data(X, Y[nt0])
    return fig


def save_sol_snap(Y, X, T, t0=.5, filename=None, name=None,
                  ylims=None, labs=None, plott=None):
    """Wrapper for sol_frame, saves the frame to file specified.

    Parameters
    ----------
    name :
        name of the solution e.g. name of the solver used (Default value = None)
    filename :
        name of the file, overrides automatic generation (Default value = None)
    Y :
        solution, array |T| x |X| x n, where n is dimension of the solution
    X :
        space interval discetization
    T :
        time interval discretization
    t0 :
        time to take snap at (Default value = .5)
    ylims :
        limits for y axis, default are 10% offsets of Y extremes
    labs :
        labels to use for parts of the solution (Default value = None)
    plott :
        plot type - how to plot data: tested plot, step (Default value = None)

    Returns
    -------
    fig
    """

    if filename is None:
        filename = "{0}_solsnap{1:3.2f}-{2:3.3}".format(name, t0, T[-1]).replace(".", "_")
        if name is None:
            name = "unknown_solver"
        filename = "{0}_solsnap{1:3.2f}-{2:3.3}".format(name, t0, T[-1]).replace(".", "_")
        filename = pjoin("semestralka", "figs", filename)

    fig = plt.figure(filename)

    snap1 = sol_frame(Y, X, T, t0=t0, ylims=ylims, labs=labs, plott=None)
    if not isinstance(Y, nm.ndarray):
        plt.plot(X, Y[0][0], label="q(x, 0)")
    else:
        if len(Y.shape) > 2:
            plt.plot(X, Y[0, :, 0], label="q(x, 0)")
        else:
            plt.plot(X, Y[0, :], label="q(x, 0)")
    plt.legend()
    snap1.savefig(filename)
    return fig


def plotsXT(Y1, Y2, YE, extent, lab1=None, lab2=None, lab3=None):
    """Plots Y1 and Y2 to one axes and YE to the second axes,
    Y1 and Y2 are presumed to be two solutions and YE their error

    Parameters
    ----------
    Y1 :
        solution 1, shape = (space nodes, time nodes)
    Y2 :
        solution 2, shape = (space nodes, time nodes)
    YE :
        soulutio 1 - soulution 2||
    extent :
        imshow extent
    lab1 :
         (Default value = None)
    lab2 :
         (Default value = None)
    lab3 :
         (Default value = None)
    """

    # >> Plot contours
    cmap1 = plt.cm.get_cmap("bwr")
    cmap1.set_bad('white')
    # cmap2 = plt.cm.get_cmap("BrBG")
    # cmap2.set_bad('white')
    bounds = nm.arange(-1, 1, .05)
    norm1 = colors.BoundaryNorm(bounds, cmap1.N)
    # norm2 = colors.BoundaryNorm(bounds, cmap2.N)

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, sharey=True)
    fig.suptitle("X-T plane plot")
    if lab1 is not None:
        ax1.set(title=lab1)
    c1 = ax1.imshow(Y1, extent=extent,
                    cmap=cmap1, norm=norm1,
                    interpolation='none',
                    origin='lower')
    ax1.grid()
    if lab2 is not None:
        ax2.set(title=lab2)
    c2 = ax2.imshow(Y2, extent=extent,
                    cmap=cmap1, norm=norm1,
                    interpolation='none',
                    origin='lower')
    ax2.grid()

    if lab3 is not None:
        ax3.set(title=lab3)
    c3 = ax3.imshow(YE, extent=extent,
                    cmap="bwr", norm=norm1,
                    interpolation='none',
                    origin='lower')
    ax3.grid()
    fig.colorbar(c3, ax=[ax1, ax2, ax3])


def load_state_1D_vtk(name):
    """Load one VTK file containing state in time

    Parameters
    ----------
    name : str

    Returns
    -------
    coors : ndarray
    u : ndarray
    """

    from sfepy.discrete.fem.meshio import MeshioLibIO
    io = MeshioLibIO(name)
    coors = io.read(Mesh()).coors[:, 0, None]
    data = io.read_data(step=0)
    var_name = head([k for k in data.keys() if "_modal" in k])[:-1]
    if var_name is None:
        print("File {} does not contain modal data.".format(name))
        return
    porder = len([k for k in data.keys() if var_name in k])


    u = nm.zeros((porder, coors.shape[0] - 1, 1, 1))
    for ii in range(porder):
        u[ii, :, 0, 0] = data[var_name+'{}'.format(ii)].data

    return coors, u


def load_1D_vtks(fold, name):
    """Reads series of .vtk files and crunches them into form
    suitable for plot10_DG_sol.

    Attempts to read modal cell data for variable mod_data. i.e.

    ``?_modal{i}``, where i is number of modal DOF

    Resulting solution data have shape:
    ``(order, nspace_steps, ntime_steps, 1)``

    Parameters
    ----------
    fold :
        folder where to look for files
    name :
        used in ``{name}.i.vtk, i = 0,1, ... tns - 1``

    Returns
    -------
    coors : ndarray
    mod_data : ndarray
        solution data

    """

    files = glob(pjoin(fold, name) + ".[0-9]*")

    if len(files) == 0: # no multiple time steps, try loading single file
        print("No files {} found in {}".format(pjoin(fold, name) + ".[0-9]*", fold))
        print("Trying {}".format(pjoin(fold, name) + ".vtk"))
        files = glob(pjoin(fold, name) + ".vtk")
        if files:
            return load_state_1D_vtk(files[0])
        else:
            print("Nothing found.")
            return

    io = MeshioLibIO(files[0])
    coors = io.read(Mesh()).coors[:, 0, None]
    data = io.read_data(step=0)
    var_name = head([k for k in data.keys() if "_modal" in k])[:-1]
    if var_name is None:
        print("File {} does not contain modal data.".format(files[0]))
        return
    porder = len([k for k in data.keys() if var_name in k])

    tn = len(files)
    nts = sorted([int(f.split(".")[-2]) for f in files])

    digs = len(files[0].split(".")[-2])
    full_name_form = ".".join((pjoin(fold, name), ("{:0" + str(digs) + "d}"), "vtk"))

    mod_data = nm.zeros((porder, coors.shape[0] - 1, tn, 1))
    for i, nt in enumerate(nts):
        io = MeshioLibIO(full_name_form.format(nt))
        # parameter "step" does nothing, but is obligatory
        data = io.read_data(step=0)
        for ii in range(porder):
            mod_data[ii, :, i, 0] = data[var_name+'{}'.format(ii)].data

    return coors, mod_data


def animate_1D_DG_sol(coors, t0, t1, u,
                      tn=None, dt=None,
                      ic=lambda x: 0.0, exact=lambda x, t: 0,
                      delay=None, polar=False):
    """Animates solution to 1D problem produced by DG:
        1. animates DOF values in elements as steps
        2. animates reconstructed solution with discontinuities

    Parameters
    ----------
    coors :
        coordinates of the mesh
    t0 : float
        starting time
    t1 : float
        final time
    u :
        vectors of DOFs, for each order one, shape(u) = (order, nspace_steps, ntime_steps, 1)
    ic :
        analytical initial condition, optional (Default value = lambda x: 0.0)
    tn :
        number of time steps to plot, starting at 0, if None and dt is not None run animation through
        all time steps, spaced dt within [t0, tn] (Default value = None)
    dt :
        time step size, if None and tn is not None computed as (t1- t0) / tn otherwise set to 1
        if dt and tn are both None, t0 and t1 are ignored and solution is animated as if in time 0 ... ntime_steps (Default value = None)
    exact :
         (Default value = lambda x)
    t: 0 :

    delay :
         (Default value = None)
    polar :
         (Default value = False)

    Returns
    -------
    anim_dofs : animation object of DOFs,
    anim_recon : animation object of reconstructed solution
    """


    # Setup space coordinates
    XN = coors[-1]
    X1 = coors[0]
    Xvol = XN - X1
    X = (coors[1:] + coors[:-1]) / 2
    XS = nm.linspace(X1, XN, 500)[:, None]

    if polar: # setup polar coorinates
        coors *= 2*nm.pi
        X *= 2*nm.pi
        XS *= 2*nm.pi

    # Setup times
    if tn is not None and dt is not None:
        T = nm.array(nm.cumsum(nm.ones(tn) * dt))
    elif tn is not None:
        T, dt = nm.linspace(t0, t1, tn, retstep=True)
    elif dt is not None:
        tn = int(nm.ceil(float(t1 - t0) / dt))
        T = nm.linspace(t0, t1, tn)
    else:
        T = nm.arange(nm.shape(u)[2])

    n_nod = len(coors)
    n_el_nod = nm.shape(u)[0]
    # prepend u[:, 0, ...] to all time frames for plotting step in left corner
    u_step = nm.append(u[:, 0:1, :, 0], u[:, :, :, 0], axis=1)

    # Plot DOFs directly
    figs = plt.figure()
    if polar:
        axs = plt.subplot(111, projection='polar')
        axs.set_theta_direction('clockwise')

    else:
        axs = plt.subplot(111)

    # Plot mesh
    axs.vlines(coors[:, 0], ymin=0, ymax=.5, colors="grey")
    axs.vlines((X1, XN), ymin=0, ymax=.5, colors="k")
    axs.vlines(X, ymin=0, ymax=.3, colors="grey", linestyles="--")

    axs.plot([X1, XN], [1, 1], 'k')

    # Plot IC and its sampling
    for i in range(n_el_nod):
        c0 = axs.plot(X, u[i, :, 0, 0],
                      label="IC-{}".format(i),
                      marker=".", ls="")[0].get_color()
        # c1 = plt.plot(X, u[1, :, 0, 0], label="IC-1", marker=".", ls="")[0].get_color()
        # # plt.plot(coors, .1*alones(n_nod), marker=".", ls="")
        axs.step(coors[1:], u[i, :, 0,  0], color=c0)
        # plt.step(coors[1:], u[1, :, 0,  0], color=c1)
        # plt.plot(coors[1:], sic[1, :], label="IC-1", color=c1)
    if ic is not None:
        ics = ic(XS)
        axs.plot(nm.squeeze(XS), nm.squeeze(ics), label="IC-ex")

    # Animate sampled solution DOFs directly
    anim_dofs = animate1D_dgsol(u_step.T, coors, T, axs, figs,
                                ylims=[-1, 2],
                                plott="step",
                                delay=delay)
    if not polar:
        axs.set_xlim(coors[0] - .1 * Xvol, coors[-1] + .1 * Xvol)
    axs.legend(loc="upper left")
    axs.set_title("Sampled solution")


    # Plot reconstructed solution
    figr = plt.figure()
    if polar:
        axr = plt.subplot(111, projection='polar')
        axr.set_theta_direction('clockwise')
    else:
        axr = plt.subplot(111)

    # Plot mesh
    axr.vlines(coors[:, 0], ymin=0, ymax=.5, colors="grey")
    axr.vlines((X1, XN), ymin=0, ymax=.5, colors="k")
    axr.vlines(X, ymin=0, ymax=.3, colors="grey", linestyles="--")

    axr.plot([X1, XN], [1, 1], 'k')

    # Plot discontinuously!
    # (order, space_steps, t_steps, 1)
    ww, xx = reconstruct_legendre_dofs(coors, tn, u)
    # plt.vlines(xx, ymin=0, ymax=.3, colors="green")

    # plot reconstructed IC
    axr.plot(xx, ww[:, 0], label="IC")

    # get exact solution values
    if exact is not None:
        exact_vals = exact(xx, T)[..., None]
        labs = ["q{}(x,t)".format(i) for i in range(ww.shape[-1])] + ["exact"]
        ww = nm.concatenate((ww, exact_vals), axis=-1)
    else:
        labs = None

    # Animate reconstructed solution
    anim_recon = animate1D_dgsol(ww.swapaxes(0, 1), xx, T, axr, figr,
                                 ylims=[-1, 2],
                                 labs=labs,
                                 delay=delay)
    if not polar:
        axr.set_xlim(coors[0] - .1 * Xvol, coors[-1] + .1 * Xvol)
    axr.legend(loc="upper left")
    axr.set_title("Reconstructed solution")

    # sol_frame(u[:, :, :, 0].T, nm.append(coors, coors[-1]), T, t0=0., ylims=[-1, 1], plott="step")
    plt.show()
    return anim_dofs, anim_recon


def plot1D_legendre_dofs(coors, dofss, fun=None):
    """Plots values of DOFs as steps

    Parameters
    ----------
    coors :
        coordinates of nodes of the mesh
    dofss :
        iterable of different projections' DOFs into legendre space
    fun :
        analytical function to plot (Default value = None)
    """
    X = (coors[1:] + coors[:-1]) / 2
    plt.figure("DOFs for function fun")
    plt.gcf().clf()
    for ii, dofs in enumerate(dofss):
        for i in range(dofs.shape[1]):
            c0 = plt.plot(X, dofs[:, i], label="fun-{}dof-{}".format(ii, i), marker=".", ls="")[0].get_color()
            # # plt.plot(coors, .1*alones(n_nod), marker=".", ls="")
            plt.step(coors[1:], dofs[:, i], color=c0)
            # plt.plot(coors[1:], sic[1, :], label="IC-1", color=c1)

    if fun is not None:
        xs = nm.linspace(nm.min(coors), nm.max(coors), 500)[:, None]
        plt.plot(xs, fun(xs), label="fun-ex")
    plt.legend()
    # plt.show()


def reconstruct_legendre_dofs(coors, tn, u):
    """Creates solution and coordinates vector which when plotted as

        plot(xx, ww)

    represent solution reconstructed from DOFs in Legendre poly space at
    cell borders.

    Works only as linear interpolation between cell boundary points

    Parameters
    ----------
    coors :
        coors of nodes of the mesh
    u :
        vectors of DOFs, for each order one,
        shape(u) = (order, nspace_steps, ntime_steps, 1)
    tn :
        number of time steps to reconstruct,
        if None all steps are reconstructed

    Returns
    -------
    ww : ndarray
        solution values vector, shape is (3 * nspace_steps - 1, ntime_steps, 1),
    xx : ndarray
        corresponding coordinates vector, shape is (3 * nspace_steps - 1, 1)
    """

    XN = coors[-1]
    X1 = coors[0]
    n_nod = len(coors) - 1
    if tn is None:
        tn = nm.shape(u)[2]
    n_el_nod = nm.shape(u)[0]

    ww = nm.zeros((3 * n_nod - 1, tn, 1))

    for i in range(n_el_nod):
        ww[0:-1:3] = ww[0:-1:3] + (-1)**i * u[i, :, :]  # left edges of elements
        ww[1::3] = ww[1::3] + u[i, :, :]  # right edges of elements
    # Nans ensure plotting of discontinuities at element borders
    ww[2::3, :] = nm.nan

    # nodes for plotting reconstructed solution
    xx = nm.zeros((3 * n_nod - 1, 1))
    xx[0] = X1
    xx[-1] = XN
    # the ending is still a bit odd, but hey, it works!
    xx[1:-1] = nm.repeat(coors[1:-1], 3)[:, None]
    return ww, xx
