"Jones and Mueller Calculus"
from prysm.mathops import np
from prysm.conf import config

def _empty_jones(shape=None):
    """Returns an empty array to populate with jones matrix elements.

    Parameters
    ----------
    shape : list
        shape to prepend to the jones matrix array. shape = [32,32] returns an array of shape [32,32,2,2]
        where the matrix is assumed to be in the last indices. Defaults to None, which returns a 2x2 array.

    Returns
    -------
    numpy.ndarray
        The empty array of specified shape
    """

    if shape is None:

        shape = (2,2)

    else:

        shape = (*shape,2,2)

    return np.zeros(shape,dtype=config.precision_complex)


def jones_rotation_matrix(theta,shape=None):
    """A rotation matrix for rotating the coordinate system transverse to propagation.
    source: https://en.wikipedia.org/wiki/Rotation_matrix.

    Parameters
    ----------
    theta : float
        angle in radians to rotate the jones matrix with respect to the x-axis.

    shape : list
        shape to prepend to the jones matrix array. shape = [32,32] returns an array of shape [32,32,2,2]
        where the matrix is assumed to be in the last indices. Defaults to None, which returns a 2x2 array.

    Returns
    -------
    numpy.ndarray
        2D rotation matrix
    """

    jones = _empty_jones(shape=shape)
    cost = np.cos(theta)
    sint = np.sin(theta)
    jones[...,0,0] = cost
    jones[...,0,1] = sint
    jones[...,1,0] = -sint
    jones[...,1,1] = cost

    return jones

def linear_retarder(retardance,theta=0,shape=None):
    """Generates a homogenous linear retarder jones matrix.

    Parameters
    ----------
    retardance : float
        phase delay experienced by the slow state in radians.

    theta : float
        angle in radians the linear retarder is rotated with respect to the x-axis.
        Defaults to 0.

    shape : list
        shape to prepend to the jones matrix array. shape = [32,32] returns an array of shape [32,32,2,2]
        where the matrix is assumed to be in the last indices. Defaults to None, which returns a 2x2 array.


    Returns
    -------
    retarder : numpy.ndarray
        numpy array containing the retarder matrices
    """

    retphasor = np.exp(1j*retardance)

    jones = _empty_jones(shape=shape)

    jones[...,0,0] = 1
    jones[...,1,1] = retphasor

    retarder = jones_rotation_matrix(-theta) @ jones @ jones_rotation_matrix(theta)

    return retarder

def linear_diattenuator(alpha,theta=0,shape=None):
    """Generates a homogenous linear diattenuator jones matrix.

    Parameters
    ----------
    alpha : float
        Fraction of the light that passes through the partially transmitted channel. 
        If 1, this is an unpolarizing plate. If 0, this is a perfect polarizer

    theta : float
        angle in radians the linear retarder is rotated with respect to the x-axis.
        Defaults to 0.

    shape : list
        shape to prepend to the jones matrix array. shape = [32,32] returns an array of shape [32,32,2,2]
        where the matrix is assumed to be in the last indices. Defaults to None, which returns a 2x2 array.


    Returns
    -------
    diattenuator : numpy.ndarray
        numpy array containing the diattenuator matrices
    """
    assert (alpha >= 0) and (alpha <= 1), f"alpha cannot be less than 0 or greater than 1, got: {alpha}"  

    jones = _empty_jones(shape=shape)
    jones[...,0,0] = 1
    jones[...,1,1] = alpha

    diattenuator = jones_rotation_matrix(-theta) @ jones @ jones_rotation_matrix(theta)

    return diattenuator

def half_wave_plate(theta=0,shape=None):
    """Make a half wave plate jones matrix. Just a wrapper for linear_retarder.

    Parameters
    ----------
    theta : float
        angle in radians the linear retarder is rotated with respect to the x-axis.
        Defaults to 0.
    shape : list
        shape to prepend to the jones matrix array. shape = [32,32] returns an array of shape [32,32,2,2]
        where the matrix is assumed to be in the last indices. Defaults to None, which returns a 2x2 array.

    Returns
    -------
    linear_retarder
        a linear retarder with half-wave retardance
    """
    return linear_retarder(np.pi,theta=theta,shape=shape)

def quarter_wave_plate(theta=0,shape=None):
    """Make a quarter wave plate jones matrix. Just a wrapper for linear_retarder.

    Parameters
    ----------
    theta : float
        angle in radians the linear retarder is rotated with respect to the x-axis.
        Defaults to 0.
    shape : list, optional
        shape to prepend to the jones matrix array. shape = [32,32] returns an array of shape [32,32,2,2]
        where the matrix is assumed to be in the last indices. Defaults to None, which returns a 2x2 array.

    Returns
    -------
    linear_retarder
        a linear retarder with quarter-wave retardance
    """
    return linear_retarder(np.pi/2,theta=theta,shape=shape)

def linear_polarizer(theta=0,shape=None):
    """Make a linear polarizer jones matrix. Just a wrapper for linear_diattenuator.

    Returns
    -------
    theta : float
        angle in radians the linear retarder is rotated with respect to the x-axis.
        Defaults to 0.
    shape : list
        shape to prepend to the jones matrix array. shape = [32,32] returns an array of shape [32,32,2,2]
        where the matrix is assumed to be in the last indices. Defaults to None, which returns a 2x2 array.

    Returns
    -------
    linear_diattenuator
        a linear diattenuator with unit diattenuation
    """

    return linear_diattenuator(0,theta=theta,shape=shape)

def jones_to_mueller(jones):
    """Construct a Mueller Matrix given a Jones Matrix. From Chipman, Lam, and Young Eq (6.99).

    Parameters
    ----------
    jones : ndarray with final dimensions 2x2
        The complex-valued jones matrices to convert into mueller matrices

    Returns
    -------
    M : np.ndarray
        Mueller matrix
    """

    U = np.array([[1,0,0,1],
                  [1,0,0,-1],
                  [0,1,1,0],
                  [0,1j,-1j,0]])/np.sqrt(2)

    jprod = np.kron(np.conj(jones),jones)
    M = np.real(U @ jprod @ np.linalg.inv(U))

    return M

def pauli_spin_matrix(index,shape=None):
    """Generates a pauli spin matrix used for Jones matrix data reduction. From CLY Eq 6.108.

    Parameters
    ----------
    index : int
        0 - returns the identity matrix
        1 - returns a linear half-wave retarder oriented horizontally
        2 - returns a linear half-wave retarder oriented 45 degrees
        3 - returns a circular half-wave retarder
    shape : list, optional
        shape to prepend to the jones matrix array. shape = [32,32] returns an array of shape [32,32,2,2]
        where the matrix is assumed to be in the last indices. Defaults to None, which returns a 2x2 array. by default None

    Returns
    -------
    jones
        pauli spin matrix of index specified
    """

    jones = _empty_jones(shape=shape)

    assert index in (0,1,2,3), f"index should be 0,1,2, or 3. Got {index}"

    if index == 0:
        jones[...,0,0] = 1
        jones[...,1,1] = 1

    elif index == 1:
        jones[...,0,0] = 1
        jones[...,1,1] = -1

    elif index == 2:
        jones[...,0,1] = 1
        jones[...,1,0] = 1

    elif index == 3:
        jones[...,0,1] = -1j
        jones[...,1,0] = 1j

    return jones

def pauli_coefficients(jones):
    """Compute the pauli coefficients of a jones matrix.

    Parameters
    ----------
    jones : numpy.ndarray  
        complex jones matrix to decompose


    Returns
    -------
    c0,c1,c2,c3
        complex coefficients of pauli matrices
    """

    c0 = (jones[...,0,0] + jones[...,1,1])/2
    c1 = (jones[...,0,0] - jones[...,1,1])/2
    c2 = (jones[...,0,1] + jones[...,1,0])/2
    c3 = 1j*(jones[...,0,1] - jones[...,1,0])/2

    return c0,c1,c2,c3

def jones_adapter(wavefunction,prop_func,*prop_func_args,**prop_func_kwargs):
    """wrapper for propagation functions to accomodate jones wavefronts

    Parameters
    ----------
    wavefunction : numpy.ndarray
        generally complex wavefunction of shape N x M x 2 x 2, where N and M are the spatial dimensions
        and the last two dimensions hold the jones matrix for each spatial dimension
    prop_func : function
        function of the prysm.propagation module

    Returns
    -------
    out : numpy.ndarray
        complex wavefunction propagated using prop_func of the same shape and dtype of wavefunction
    """

    # Treat Wavefront.func
    if hasattr(prop_func,'__self__'):
        func = getattr(prop_func.__class__,func.__name__)

    # Treat prysm.propagation.func
    elif prop_func.__class__.__module__ == 'builtins':

        J00 = wavefunction[...,0,0]
        J01 = wavefunction[...,0,1]
        J10 = wavefunction[...,1,0]
        J11 = wavefunction[...,1,1]
        tmp = []
        for E in [J00, J01, J10, J11]:
            ret = prop_func(E, *prop_func_args, **prop_func_kwargs)
            tmp.append(ret)
    
    # one path, return list (no extra copies/allocs)
    # return tmp

    # different path, pack it back in (waste copies)
    out = np.empty((*tmp[0].shape,2,2), tmp[0].dtype)
    out[...,0,0] = tmp[0]
    out[...,0,1] = tmp[1]
    out[...,1,0] = tmp[2]
    out[...,1,1] = tmp[3]

    # return in original format
    return out
