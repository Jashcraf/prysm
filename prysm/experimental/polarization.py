"Jones and Mueller Calculus"
import numpy as np

def _empty_jones(shape=None):

    """returns an empty array to populate with jones matrix elements

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

        shape = [2,2]

    else:

        shape.append(2)
        shape.append(2)

    return np.zeros(shape,dtype='complex128')


def jones_rotation_matrix(theta,shape=None):
    """a rotation matrix for rotating the coordinate system transverse to propagation.
    source: https://en.wikipedia.org/wiki/Rotation_matrix

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
    jones[...,0,0] = np.cos(theta)
    jones[...,0,1] = np.sin(theta)
    jones[...,1,0] = -np.sin(theta)
    jones[...,1,1] = np.cos(theta)

    return jones

def linear_retarder(retardance,theta=0,shape=None):

    """generates a homogenous linear retarder jones matrix

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

    """generates a homogenous linear diattenuator jones matrix

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
    return linear_retarder(np.pi,theta=theta,shape=shape)

def quarter_wave_plate(theta=0,shape=None):
    return linear_retarder(np.pi/2,theta=theta,shape=shape)

def linear_polarizer(theta=0,shape=None):
    return linear_diattenuator(0,theta=theta,shape=shape)

def jones_to_mueller(jones):

    U = np.array([[],[],[],[]])
