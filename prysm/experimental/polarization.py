"Jones and Mueller Calculus"
from prysm.mathops import np, fft, is_odd
from prysm.conf import config
from prysm.propagation import Wavefront
from prysm.fttools import pad2d, crop_center, mdft, czt
import functools

"""Numerical optical propagation."""
import copy
import numbers
import inspect
import operator
import warnings
from math import ceil
from collections.abc import Iterable


def broadcast_kron(a,b):
    """broadcasted kronecker product of two N,M,...,2,2 arrays. Used for jones -> mueller conversion
    In the unbroadcasted case, this output looks like

    out = [a[0,0]*b,a[0,1]*b]
          [a[1,0]*b,a[1,1]*b]

    where out is a N,M,...,4,4 array. I wrote this to work for generally shaped kronecker products where the matrix
    is contained in the last two axes, but it's only tested for the Nx2x2 case

    Parameters
    ----------
    a : numpy.ndarray
        N,M,...,2,2 array used to scale b in kronecker product
    b : numpy.ndarray
        N,M,...,2,2 array used to form block matrices in kronecker product

    Returns
    -------
    out
        N,M,...,4,4 array
    """

    return np.einsum('...ik,...jl',a,b).reshape([*a.shape[:-2],int(a.shape[-2]*b.shape[-2]),int(a.shape[-1]*b.shape[-1])])

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
    
    if jones.ndim == 2:
        jprod = np.kron(np.conj(jones),jones)
    else:
        # broadcasted kronecker product with einsum
        jprod = broadcast_kron(np.conj(jones),jones)
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

def jones_decorator(prop_func):

    @functools.wraps(prop_func)
    def wrapper(*args,**kwargs):

        
        # this is a function
        wavefunction = args[0]
        if len(args) > 1:
            other_args = args[1:]
        else:
            other_args = ()

        if wavefunction.ndim == 2:
            # pass through non-jones case
            return prop_func(*args,**kwargs)

        J00 = wavefunction[...,0,0]
        J01 = wavefunction[...,0,1]
        J10 = wavefunction[...,1,0]
        J11 = wavefunction[...,1,1]
        tmp = []
        for E in [J00, J01, J10, J11]:
            ret = prop_func(E, *other_args, **kwargs)
            tmp.append(ret)
        
        out = np.empty([*ret.shape,2,2],dtype=ret.dtype)
        out[...,0,0] = tmp[0]
        out[...,0,1] = tmp[1]
        out[...,1,0] = tmp[2]
        out[...,1,1] = tmp[3]
        
        return out
    
    return wrapper

@jones_decorator
def focus(wavefunction, Q):
    """Propagate a pupil plane to a PSF plane.

    Parameters
    ----------
    wavefunction : numpy.ndarray
        the pupil wavefunction
    Q : float
        oversampling / padding factor

    Returns
    -------
    psf : numpy.ndarray
        point spread function

    """
    if Q != 1:
        padded_wavefront = pad2d(wavefunction, Q)
    else:
        padded_wavefront = wavefunction

    impulse_response = fft.fftshift(fft.fft2(fft.ifftshift(padded_wavefront), norm='ortho'))
    return impulse_response

@jones_decorator
def unfocus(wavefunction, Q):
    """Propagate a PSF plane to a pupil plane.

    Parameters
    ----------
    wavefunction : numpy.ndarray
        the pupil wavefunction
    Q : float
        oversampling / padding factor

    Returns
    -------
    pupil : numpy.ndarray
        field in the pupil plane

    """
    if Q != 1:
        padded_wavefront = pad2d(wavefunction, Q)
    else:
        padded_wavefront = wavefunction

    return fft.fftshift(fft.ifft2(fft.ifftshift(padded_wavefront), norm='ortho'))

@jones_decorator
def focus_fixed_sampling(wavefunction, input_dx, prop_dist,
                         wavelength, output_dx, output_samples,
                         shift=(0, 0), method='mdft'):
    """Propagate a pupil function to the PSF plane with fixed sampling.

    Parameters
    ----------
    wavefunction : numpy.ndarray
        the pupil wavefunction
    input_dx : float
        spacing between samples in the pupil plane, millimeters
    prop_dist : float
        propagation distance along the z distance
    wavelength : float
        wavelength of light
    output_dx : float
        sample spacing in the output plane, microns
    output_samples : int
        number of samples in the square output array
    shift : tuple of float
        shift in (X, Y), same units as output_dx
    method : str, {'mdft', 'czt'}
        how to propagate the field, matrix DFT or Chirp Z transform
        CZT is usually faster single-threaded and has less memory consumption
        MDFT is usually faster multi-threaded and has more memory consumption

    Returns
    -------
    data : numpy.ndarray
        2D array of data

    """
    if not isinstance(output_samples, Iterable):
        output_samples = (output_samples, output_samples)

    dia = wavefunction.shape[0] * input_dx
    Q = Q_for_sampling(input_diameter=dia,
                       prop_dist=prop_dist,
                       wavelength=wavelength,
                       output_dx=output_dx)
    if shift[0] != 0 or shift[1] != 0:
        shift = (shift[0]/output_dx, shift[1]/output_dx)

    if method == 'mdft':
        out = mdft.dft2(ary=wavefunction, Q=Q, samples_out=output_samples, shift=shift)
    elif method == 'czt':
        out = czt.czt2(ary=wavefunction, Q=Q, samples_out=output_samples, shift=shift)

    return out

@jones_decorator
def focus_fixed_sampling_backprop(wavefunction, input_dx, prop_dist,
                                  wavelength, output_dx, output_samples,
                                  shift=(0, 0), method='mdft'):
    """Propagate a pupil function to the PSF plane with fixed sampling.

    Parameters
    ----------
    wavefunction : numpy.ndarray
        the pupil wavefunction
    input_dx : float
        spacing between samples in the pupil plane, millimeters
    prop_dist : float
        propagation distance along the z distance
    wavelength : float
        wavelength of light
    output_dx : float
        sample spacing in the output plane, microns
    output_samples : int
        number of samples in the square output array
    shift : tuple of float
        shift in (X, Y), same units as output_dx
    method : str, {'mdft', 'czt'}
        how to propagate the field, matrix DFT or Chirp Z transform
        CZT is usually faster single-threaded and has less memory consumption
        MDFT is usually faster multi-threaded and has more memory consumption

    Returns
    -------
    data : numpy.ndarray
        2D array of data

    """
    if not isinstance(output_samples, Iterable):
        output_samples = (output_samples, output_samples)

    dia = output_samples[0] * input_dx
    Q = Q_for_sampling(input_diameter=dia,
                       prop_dist=prop_dist,
                       wavelength=wavelength,
                       output_dx=output_dx)
    if shift[0] != 0 or shift[1] != 0:
        shift = (shift[0]/output_dx, shift[1]/output_dx)

    if method == 'mdft':
        out = mdft.dft2_backprop(wavefunction, Q, samples_in=output_samples, shift=shift)
    elif method == 'czt':
        raise ValueError('gradient backpropagation not yet implemented for CZT')
        out = czt.czt2_backprop(ary=wavefunction, Q=Q, samples=output_samples, shift=shift)

    return out

@jones_decorator
def unfocus_fixed_sampling(wavefunction, input_dx, prop_dist,
                           wavelength, output_dx, output_samples,
                           shift=(0, 0), method='mdft'):
    """Propagate an image plane field to the pupil plane with fixed sampling.

    Parameters
    ----------
    wavefunction : numpy.ndarray
        the image plane wavefunction
    input_dx : float
        spacing between samples in the pupil plane, millimeters
    prop_dist : float
        propagation distance along the z distance
    wavelength : float
        wavelength of light
    output_dx : float
        sample spacing in the output plane, microns
    output_samples : int
        number of samples in the square output array
    shift : tuple of float
        shift in (X, Y), same units as output_dx
    method : str, {'mdft', 'czt'}
        how to propagate the field, matrix DFT or Chirp Z transform
        CZT is usually faster single-threaded and has less memory consumption
        MDFT is usually faster multi-threaded and has more memory consumption

    Returns
    -------
    x : numpy.ndarray
        x axis unit, 1D ndarray
    y : numpy.ndarray
        y axis unit, 1D ndarray
    data : numpy.ndarray
        2D array of data

    """
    # we calculate sampling parameters
    # backwards so we can reuse as much code as possible
    if not isinstance(output_samples, Iterable):
        output_samples = (output_samples, output_samples)

    dias = [output_dx * s for s in output_samples]
    dia = max(dias)
    Q = Q_for_sampling(input_diameter=dia,
                       prop_dist=prop_dist,
                       wavelength=wavelength,
                       output_dx=input_dx)  # not a typo

    Q /= wavefunction.shape[0] / output_samples[0]

    if shift[0] != 0 or shift[1] != 0:
        shift = (shift[0]/output_dx, shift[1]/output_dx)

    if method == 'mdft':
        out = mdft.idft2(ary=wavefunction, Q=Q, samples_out=output_samples, shift=shift)
    elif method == 'czt':
        out = czt.iczt2(ary=wavefunction, Q=Q, samples_out=output_samples, shift=shift)

    out *= Q
    return out


def Q_for_sampling(input_diameter, prop_dist, wavelength, output_dx):
    """Value of Q for a given output sampling, given input sampling.

    Parameters
    ----------
    input_diameter : float
        diameter of the input array in millimeters
    prop_dist : float
        propagation distance along the z distance, millimeters
    wavelength : float
        wavelength of light, microns
    output_dx : float
        sampling in the output plane, microns

    Returns
    -------
    float
        requesite Q

    """
    resolution_element = (wavelength * prop_dist) / (input_diameter)
    return resolution_element / output_dx


def pupil_sample_to_psf_sample(pupil_sample, samples, wavelength, efl):
    """Convert pupil sample spacing to PSF sample spacing.  fλ/D or Q.

    Parameters
    ----------
    pupil_sample : float
        sample spacing in the pupil plane
    samples : int
        number of samples present in both planes (must be equal)
    wavelength : float
        wavelength of light, in microns
    efl : float
        effective focal length of the optical system in mm

    Returns
    -------
    float
        the sample spacing in the PSF plane

    """
    return (efl * wavelength) / (pupil_sample * samples)


def psf_sample_to_pupil_sample(psf_sample, samples, wavelength, efl):
    """Convert PSF sample spacing to pupil sample spacing.

    Parameters
    ----------
    psf_sample : float
        sample spacing in the PSF plane
    samples : int
        number of samples present in both planes (must be equal)
    wavelength : float
        wavelength of light, in microns
    efl : float
        effective focal length of the optical system in mm

    Returns
    -------
    float
        the sample spacing in the pupil plane

    """
    return (efl * wavelength) / (psf_sample * samples)


def fresnel_number(a, L, lambda_):
    """Compute the Fresnel number.

    Notes
    -----
    if the fresnel number is << 1, paraxial assumptions hold for propagation

    Parameters
    ----------
    a : float
        characteristic size ("radius") of an aperture
    L : float
        distance of observation
    lambda_ : float
        wavelength of light, same units as a

    Returns
    -------
    float
        the fresnel number for these parameters

    """
    return a**2 / (L * lambda_)


def talbot_distance(a, lambda_):
    """Compute the talbot distance.

    Parameters
    ----------
    a : float
        period of the grating, units of microns
    lambda_ : float
        wavelength of light, units of microns

    Returns
    -------
    float
        talbot distance, units of microns

    """
    num = lambda_
    den = 1 - np.sqrt(1 - lambda_**2/a**2)
    return num / den

@jones_decorator
def angular_spectrum(field, wvl, dx, z, Q=2, tf=None):
    """Propagate a field via the angular spectrum method.

    Parameters
    ----------
    field : numpy.ndarray
        2D array of complex electric field values
    wvl : float
        wavelength of light, microns
    z : float
        propagation distance, units of millimeters
    dx : float
        cartesian sample spacing, units of millimeters
    Q : float
        sampling factor used.  Q>=2 for Nyquist sampling of incoherent fields
    tf : numpy.ndarray
        if not None, clobbers all other arguments
        transfer function for the propagation

    Returns
    -------
    numpy.ndarray
        2D ndarray of the output field, complex

    """
    if tf is not None:
        return fft.ifft2(fft.fft2(field) * tf)

    if Q != 1:
        field = pad2d(field, Q=Q)

    transfer_function = angular_spectrum_transfer_function(field.shape, wvl, dx, z)
    forward = fft.fft2(field)
    return fft.ifft2(forward*transfer_function)

@jones_decorator
def angular_spectrum_transfer_function(samples, wvl, dx, z):
    """Precompute the transfer function of free space.

    Parameters
    ----------
    samples : int or tuple
        (y,x) or (r,c) samples in the output array
    wvl : float
        wavelength of light, microns
    dx : float
        intersample spacing, mm
    z : float
        propagation distance, mm

    Returns
    -------
    numpy.ndarray
        ndarray of shape samples containing the complex valued transfer function
        such that X = fft2(x); xhat = ifft2(X*tf) is signal x after free space propagation

    """
    if isinstance(samples, int):
        samples = (samples, samples)

    wvl = wvl / 1e3
    ky, kx = (fft.fftfreq(s, dx).astype(config.precision) for s in samples)
    kxx = kx * kx
    kyy = ky * ky
    kyy = np.broadcast_to(ky, samples).swapaxes(0, 1)
    kxx = np.broadcast_to(kx, samples)

    return np.exp(-1j * np.pi * wvl * z * (kxx + kyy))

@jones_decorator
def to_fpm_and_back(wavefunction, wavefunction_dx, efl, fpm, fpm_dx, wavelength, method='mdft', shift=(0, 0), return_more=False):
    """Propagate to a focal plane mask, apply it, and return.
    This routine handles normalization properly for the user.
    To invoke babinet's principle, simply use to_fpm_and_back(fpm=1 - fpm).
    Parameters
    ----------
    wavefunction : numpy.ndarray
        field before the focal plane to propagate
    wavefunction_dx : float
        sampling increment in the wavefunction ,  mm;
        do not need to pass if wavefunction is a Wavefront
    efl : float
        focal length for the propagation
    fpm : numpy.ndarray
        the focal plane mask
    fpm_dx : float
        sampling increment in the focal plane,  microns;
        do not need to pass if fpm is a Wavefront
    wavelength : float
        wavelength of light, microns;
        do not need to pass if wavefunction is a Wavefront
    method : str, {'mdft', 'czt'}, optional
        how to propagate the field, matrix DFT or Chirp Z transform
        CZT is usually faster single-threaded and has less memory consumption
        MDFT is usually faster multi-threaded and has more memory consumption
    shift : tuple of float, optional
        shift in the image plane to go to the FPM
        appropriate shift will be computed returning to the pupil
    return_more : bool, optional
        if True, return (new_wavefront, field_at_fpm, field_after_fpm)
        else return new_wavefront
    Returns
    -------
    numpy.ndarray
        field after fpm
    """
    
    if wavefunction or wavelength is None:
        raise ValueError('wavefunction is not a wavefront and wvfn_dx or wavelength are none')
    fpm_samples = fpm.shape
    input_samples = wavefunction.shape
    data = wavefunction
    input_diameters = [wavefunction_dx * s for s in input_samples]
    Q_forward = [Q_for_sampling(d, efl, wavelength, fpm_dx) for d in input_diameters]

    # soummer notation: use m, which would be 0.5 for a 2x zoom
    # BDD notation: Q, would be 2 for a 2x zoom
    m_forward = [1/q for q in Q_forward]
    m_reverse = [b/a*m for a, b, m in zip(input_samples, fpm_samples, m_forward)]
    Q_reverse = [1/m for m in m_reverse]
    shift_forward = tuple(s/fpm_dx for s in shift)

    # prop forward
    kwargs = dict(ary=data, Q=Q_forward, samples_out=fpm_samples, shift=shift_forward)
    if method == 'mdft':
        field_at_fpm = mdft.dft2(**kwargs)
    elif method == 'czt':
        field_at_fpm = czt.czt2(**kwargs)

    field_after_fpm = field_at_fpm * fpm

    # shift_reverse = tuple(-s for s, q in zip(shift_forward, Q_forward))
    shift_reverse = shift_forward
    kwargs = dict(ary=field_after_fpm, Q=Q_reverse, samples_out=input_samples, shift=shift_reverse)
    if method == 'mdft':
        field_at_next_pupil = mdft.idft2(**kwargs)
    elif method == 'czt':
        field_at_next_pupil = czt.iczt2(**kwargs)

    # scaling
    # TODO: make this handle anamorphic transforms properly
    if Q_forward[0] != Q_forward[1]:
        warnings.warn(f'Forward propagation had Q {Q_forward} which was not uniform between axes, scaling is off')
    if input_samples[0] != input_samples[1]:
        warnings.warn(f'Forward propagation had input shape {input_samples} which was not uniform between axes, scaling is off')
    if fpm_samples[0] != fpm_samples[1]:
        warnings.warn(f'Forward propagation had fpm shape {fpm_samples} which was not uniform between axes, scaling is off')
    # Q_reverse is calculated from Q_forward; if one is consistent the other is

    return field_at_next_pupil

@jones_decorator
def to_fpm_and_back_backprop(wavefunction, wavefunction_dx, efl, fpm, fpm_dx, wavelength, method='mdft', shift=(0, 0), return_more=False):
    """Propagate to a focal plane mask, apply it, and return.
    This routine handles normalization properly for the user.
    To invoke babinet's principle, simply use to_fpm_and_back(fpm=1 - fpm).
    Parameters
    ----------
    wavefunction : numpy.ndarray
        field before the focal plane to propagate
    wavefunction_dx : float
        sampling increment in the wavefunction ,  mm;
        do not need to pass if wavefunction is a Wavefront
    efl : float
        focal length for the propagation
    fpm : numpy.ndarray
        the focal plane mask
    fpm_dx : float
        sampling increment in the focal plane,  microns;
        do not need to pass if fpm is a Wavefront
    wavelength : float
        wavelength of light, microns;
        do not need to pass if wavefunction is a Wavefront
    method : str, {'mdft', 'czt'}, optional
        how to propagate the field, matrix DFT or Chirp Z transform
        CZT is usually faster single-threaded and has less memory consumption
        MDFT is usually faster multi-threaded and has more memory consumption
    shift : tuple of float, optional
        shift in the image plane to go to the FPM
        appropriate shift will be computed returning to the pupil
    return_more : bool, optional
        if True, return (new_wavefront, field_at_fpm, field_after_fpm)
        else return new_wavefront
    Returns
    -------
    numpy.ndarray
        field after fpm
    """
    
    fpm_samples = fpm.shape

    # do not take complex conjugate of reals (no-op, but numpy still does it)
    if np.iscomplexobj(fpm.dtype):
        fpm = fpm.conj()

    input_samples = wavefunction.shape
    input_diameters = [wavefunction_dx * s for s in input_samples]
    Q_forward = [Q_for_sampling(d, efl, wavelength, fpm_dx) for d in input_diameters]
    # soummer notation: use m, which would be 0.5 for a 2x zoom
    # BDD notation: Q, would be 2 for a 2x zoom
    m_forward = [1/q for q in Q_forward]
    m_reverse = [b/a*m for a, b, m in zip(input_samples, fpm_samples, m_forward)]
    Q_reverse = [1/m for m in m_reverse]
    shift_forward = tuple(s/fpm_dx for s in shift)

    kwargs = dict(fbar=wavefunction_dx, Q=Q_reverse, samples_in=fpm_samples, shift=shift_forward)
    if method == 'mdft':
        Ebbar = -(mdft.idft2_backprop(**kwargs))
    elif method == 'czt':
        raise ValueError('CZT backprop not yet implemented')
        field_at_fpm = czt.czt2_backprop(**kwargs)

    intermediate = Ebbar * fpm

    kwargs = dict(fbar=intermediate, Q=Q_forward, samples_in=input_samples, shift=shift_forward)
    if method == 'mdft':
        Eabar = mdft.dft2_backprop(**kwargs)
    elif method == 'czt':
        raise ValueError('CZT backprop not yet implemented')
        field_at_next_pupil = czt.iczt2(**kwargs)

    # scaling
    # TODO: make this handle anamorphic transforms properly
    if Q_forward[0] != Q_forward[1]:
        warnings.warn(f'Forward propagation had Q {Q_forward} which was not uniform between axes, scaling is off')
    if input_samples[0] != input_samples[1]:
        warnings.warn(f'Forward propagation had input shape {input_samples} which was not uniform between axes, scaling is off')
    if fpm_samples[0] != fpm_samples[1]:
        warnings.warn(f'Forward propagation had fpm shape {fpm_samples} which was not uniform between axes, scaling is off')
    # Q_reverse is calculated from Q_forward; if one is consistent the other is

    return Eabar

@jones_decorator
def babinet(efl, lyot, fpm, fpm_dx=None, method='mdft', return_more=False):
    """Propagate through a Lyot-style coronagraph using Babinet's principle.
    This routine handles normalization properly for the user.
    Parameters
    ----------
    efl : float
        focal length for the propagation
    lyot : Wavefront or numpy.ndarray
        the Lyot stop; if None, equivalent to ones_like(self.data)
    fpm : Wavefront or numpy.ndarray
        1 - fpm
        one minus the focal plane mask (see Soummer et al 2007)
    fpm_dx : float
        sampling increment in the focal plane,  microns;
        do not need to pass if fpm is a Wavefront
    method : str, {'mdft', 'czt'}
        how to propagate the field, matrix DFT or Chirp Z transform
        CZT is usually faster single-threaded and has less memory consumption
        MDFT is usually faster multi-threaded and has more memory consumption
    return_more : bool
        if True, return each plane in the propagation
        else return new_wavefront
    Notes
    -----
    if the substrate's reflectivity or transmissivity is not unity, and/or
    the mask's density is not infinity, babinet's principle works as follows:
    suppose we're modeling a Lyot focal plane mask;
    rr = radial coordinates of the image plane, in lambda/d units
    mask = rr < 5  # 1 inside FPM, 0 outside (babinet-style)
    now create some scalars for background transmission and mask transmission
    tau = 0.9 # background
    tmask = 0.1 # mask
    mask = tau - tau*mask + rmask*mask
    the mask variable now contains 0.9 outside the spot, and 0.1 inside
    Returns
    -------
    Wavefront, Wavefront, Wavefront, Wavefront
        field after lyot, [field at fpm, field after fpm, field at lyot]
    """
    fpm = 1 - fpm
    if return_more:
        field, field_at_fpm, field_after_fpm = \
            self.to_fpm_and_back(efl=efl, fpm=fpm, fpm_dx=fpm_dx, method=method,
                                    return_more=return_more)
    else:
        field = self.to_fpm_and_back(efl=efl, fpm=fpm, fpm_dx=fpm_dx, method=method,
                                        return_more=return_more)
    # DOI: 10.1117/1.JATIS.7.1.019002
    # Eq. 26 with some minor differences in naming
    if not is_odd(field.data.shape[0]):
        coresub = np.roll(field.data, -1, axis=0)
    else:
        coresub = field.data

    field_at_lyot = self.data - np.flipud(coresub)

    if lyot is not None:
        field_after_lyot = lyot * field_at_lyot
    else:
        field_after_lyot = field_at_lyot

    field_at_lyot = Wavefront(field_at_lyot, self.wavelength, self.dx, self.space)
    field_after_lyot = Wavefront(field_after_lyot, self.wavelength, self.dx, self.space)

    if return_more:
        return field_after_lyot, field_at_fpm, field_after_fpm, field_at_lyot
    return field_after_lyot

@jones_decorator
def babinet_backprop(efl, lyot, fpm, fpm_dx=None, method='mdft'):
    """Propagate through a Lyot-style coronagraph using Babinet's principle.
    Parameters
    ----------
    efl : float
        focal length for the propagation
    lyot : Wavefront or numpy.ndarray
        the Lyot stop; if None, equivalent to ones_like(self.data)
    fpm : Wavefront or numpy.ndarray
        np.conj(1 - fpm)
        one minus the focal plane mask (see Soummer et al 2007)
    fpm_dx : float
        sampling increment in the focal plane,  microns;
        do not need to pass if fpm is a Wavefront
    method : str, {'mdft', 'czt'}
        how to propagate the field, matrix DFT or Chirp Z transform
        CZT is usually faster single-threaded and has less memory consumption
        MDFT is usually faster multi-threaded and has more memory consumption
    Returns
    -------
    Wavefront
        back-propagated gradient
    """
    # babinet's principle is implemented by
    # A = DFT(a)       |
    # C = A*B          |
    # c = iDFT(C)      | Cbar to Abar absorbed in to_fpm_and_back_backprop
    # d = c*L          | cbar = dbar * conj(L)
    # f = d - flip(a)  | dbar = d

    fpm = 1 - fpm

    dbar = self.data
    if lyot is not None:
        if np.iscomplexobj(lyot):
            lyot = np.conj(lyot)

        cbar = dbar * lyot
    else:
        cbar = dbar

    # minus from Ebefore minus Eafter fpm
    cbarW = Wavefront(cbar, self.wavelength, self.dx, self.space)
    abar = cbarW.to_fpm_and_back_backprop(efl=efl, fpm=fpm, fpm_dx=fpm_dx, method=method)

    if not is_odd(cbar.shape[0]):
        cbarflip = np.flipud(np.roll(cbar, -1, axis=0))

    abar.data += cbarflip
    return abar