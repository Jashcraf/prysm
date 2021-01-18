"""Various polynomials of optics."""

from prysm.mathops import np
from prysm.coordinates import optimize_xy_separable

from .jacobi import jacobi, jacobi_sequence  # NOQA
from .cheby import (  # NOQA
    cheby1, cheby1_sequence, cheby1_2d_sequence,
    cheby2, cheby2_sequence, cheby2_2d_sequence,
)
from .legendre import (  # NOQA
    legendre,
    legendre_sequence,
    legendre_2d_sequence,
)  # NOQA
from .zernike import (  # NOQA
    zernike_norm,
    zernike_nm,
    zernike_nm_sequence,
    zernikes_to_magnitude_angle,
    zernikes_to_magnitude_angle_nmkey,
    zero_separation as zernike_zero_separation,
    ansi_j_to_nm,
    nm_to_ansi_j,
    nm_to_fringe,
    nm_to_name,
    noll_to_nm,
    fringe_to_nm,
    barplot as zernike_barplot,
    barplot_magnitudes as zernike_barplot_magnitudes,
    top_n,
)
from .qpoly import (  # NOQA
    Qbfs, Qbfs_sequence,
    Qcon, Qcon_sequence,
    Q2d, Q2d_sequence,
)


def mode_1d_to_2d(mode, x, y, which='x'):
    """Expand a 1D representation of a mode to 2D.

    Notes
    -----
    You likely only want to use this function for plotting or similar, it is
    much faster to use sum_of_xy_modes to produce 2D surfaces described by
    a sum of modes which are separable in x and y.

    Parameters
    ----------
    mode : `numpy.ndarray`
        mode, representing a separable mode in X, Y along {which} axis
    x : `numpy.ndarray`
        x dimension, either 1D or 2D
    y : `numpy.ndarray`
        y dimension, either 1D or 2D
    which : `str`, {'x', 'y'}
        which dimension the mode is produced along

    Returns
    -------
    `numpy.ndarray`
        2D version of the mode

    """
    x, y = optimize_xy_separable(x, y)

    out = np.broadcast_to(mode, (x.size, y.size))
    if which.lower() == 'y':
        out = out.swapaxes(0, 1)  # broadcast_to will repeat along rows

    return out


def sum_of_xy_modes(modesx, modesy, x, y, weightsx=None, weightsy=None):
    """Weighted sum of separable x and y modes projected over the 2D aperture.

    Parameters
    ----------
    modesx : `iterable`
        sequence of x modes
    modesy : `iterable`
        sequence of y modes
    x : `numpy.ndarray`
        x points
    y : `numpy.ndarray`
        y points
    weightsx : `iterable`, optional
        weights to apply to modesx.  If None, [1]*len(modesx)
    weightsy : `iterable`, optional
        weights to apply to modesy.  If None, [1]*len(modesy)

    Returns
    -------
    `numpy.ndarray`
        modes summed over the 2D aperture

    """
    x, y = optimize_xy_separable(x, y)

    if weightsx is None:
        weightsx = [1]*len(modesx)
    if weightsy is None:
        weightsy = [1]*len(modesy)

    # apply the weights to the modes
    modesx = [m*w for m, w in zip(modesx, weightsx)]
    modesy = [m*w for m, w in zip(modesy, weightsy)]

    # sum the separable bases in 1D
    sum_x = np.zeros_like(x)
    sum_y = np.zeros_like(y)
    for m in modesx:
        sum_x += m
    for m in modesy:
        sum_y += m

    # broadcast to 2D and return
    shape = (x.size, y.size)
    sum_x = np.broadcast_to(sum_x, shape)
    sum_y = np.broadcast_to(sum_y, shape)
    return sum_x + sum_y


def hopkins(a, b, c, r, t, H):
    """Hopkins' aberration expansion.

    This function uses the "W020" or "W131" like notation, with Wabc separating
    into the a, b, c arguments.  To produce a sine term instead of cosine,
    make a the negative of the order.  In other words, for W222S you would use
    hopkins(2, 2, 2, ...) and for W222T you would use
    hopkins(-2, 2, 2, ...).

    Parameters
    ----------
    a : `int`
        azimuthal order
    b : `int`
        radial order
    c : `int`
        order in field ("H-order")
    r : `numpy.ndarray`
        radial pupil coordinate
    t : `numpy.ndarray`
        azimuthal pupil coordinate
    H : `numpy.ndarray`
        field coordinate

    Returns
    -------
    `numpy.ndarray`
        polynomial evaluated at this point

    """
    # c = "component"
    if a < 0:
        c1 = np.sin(abs(a)*t)
    else:
        c1 = np.cos(a*t)

    c2 = r ** b

    c3 = H ** c

    return c1 * c2 * c3