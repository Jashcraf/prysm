import numpy as np
import prysm.experimental.polarization as pol

def test_rotation_matrix():

    # Make a 45 degree rotation
    angle = np.pi/4
    control = 1/np.sqrt(2) * np.array([[1,1],[-1,1]])

    test = pol.jones_rotation_matrix(angle)

    np.testing.assert_allclose(control,test)

def test_linear_retarder():

    # Create a quarter-wave plate
    retardance = np.pi/2 # qwp retardance
    control = np.array([[1,0],[0,1j]]) # oriented at 0 deg

    test = pol.linear_retarder(retardance)

    np.testing.assert_allclose(control,test)

def test_linear_diattenuator():

    # Create an imperfect polarizer with a diattenuation of 0.75
    alpha = 0.5
    control = np.array([[1,0],[0,0.5]])

    test = pol.linear_diattenuator(alpha)

    np.testing.assert_allclose(control,test)

def test_half_wave_plate():

    hwp = np.array([[1,0],[0,-1]])
    test = pol.half_wave_plate(0)

    np.testing.assert_allclose(hwp,test)

def test_quarter_wave_plate():

    qwp = np.array([[1,0],[0,1j]])
    test = pol.quarter_wave_plate()

    np.testing.assert_allclose(qwp,test)

def test_linear_polarizer():

    lp = np.array([[1,0],[0,0]])
    test = pol.linear_polarizer()

    np.testing.assert_allclose(lp,test)

def test_jones_to_mueller():

    # Make a circular polarizer
    circ_pol = pol.quarter_wave_plate(theta=np.pi/4)

    mueller_test = pol.jones_to_mueller(circ_pol)/2
    mueller_circ = np.array([[1,0,0,0],
                             [0,0,0,-1],
                             [0,0,1,0],
                             [0,1,0,0]])/2

    np.testing.assert_allclose(mueller_circ,mueller_test,atol=1e-5)

def test_pauli_spin_matrix():

    p0 = np.array([[1,0],[0,1]])
    p1 = np.array([[1,0],[0,-1]])
    p2 = np.array([[0,1],[1,0]])
    p3 = np.array([[0,-1j],[1j,0]])

    np.testing.assert_allclose((p0,p1,p2,p3),
                              (pol.pauli_spin_matrix(0),
                               pol.pauli_spin_matrix(1),
                               pol.pauli_spin_matrix(2),
                               pol.pauli_spin_matrix(3)))
    
def test_jones_adapter_focus():
    """test jones adapter on propagation functions
    """

    from prysm.coordinates import make_xy_grid, cart_to_polar
    from prysm.geometry import circle
    from prysm.propagation import focus
    from prysm.polynomials import hopkins
    from prysm.experimental.polarization import jones_adapter
    from prysm.conf import config

    N,M = 256,256
    wvl = 1e-6

    # set up a wave function for the on-diagonals
    x,y = make_xy_grid(N,diameter=2)
    r,t = cart_to_polar(x,y)
    rho = r/5
    phi = hopkins(0,4,0,rho,t,1)
    A = circle(1,r)

    wavefunction = A*np.exp(1j*2*np.pi/wvl*phi)

    # Set up jones data, numpy.ndarray of shape N,M,2,2
    # This is essentially a non-polarizing system
    jones = np.zeros([N,M,2,2],dtype=config.precision_complex) # this represents our "wavefunction"
    jones[...,0,0] = wavefunction
    jones[...,1,1] = wavefunction

    # test focus
    jones_focus = jones_adapter(jones,focus,2)
    ref_focus = focus(wavefunction,2)

    np.testing.assert_allclose((jones_focus[...,0,0],jones_focus[...,1,1]),(ref_focus,ref_focus))

def test_jones_adapter_unfocus():
    """test jones adapter on propagation functions
    """

    from prysm.coordinates import make_xy_grid, cart_to_polar
    from prysm.geometry import circle
    from prysm.propagation import focus,unfocus
    from prysm.polynomials import hopkins
    from prysm.experimental.polarization import jones_adapter
    from prysm.conf import config

    N,M = 256,256
    wvl = 1e-6

    # set up a wave function for the on-diagonals
    x,y = make_xy_grid(N,diameter=2)
    r,t = cart_to_polar(x,y)
    rho = r/5
    phi = hopkins(0,4,0,rho,t,1)
    A = circle(1,r)

    wavefunction = A*np.exp(1j*2*np.pi/wvl*phi)
    wavefunction = focus(wavefunction,1)

    # Set up jones data, numpy.ndarray of shape N,M,2,2
    # This is essentially a non-polarizing system
    jones = np.zeros([N,M,2,2],dtype=config.precision_complex) # this represents our "wavefunction"
    jones[...,0,0] = wavefunction
    jones[...,1,1] = wavefunction

    # test focus
    jones_focus = jones_adapter(jones,unfocus,2)
    ref_focus = unfocus(wavefunction,2)

    np.testing.assert_allclose((jones_focus[...,0,0],jones_focus[...,1,1]),(ref_focus,ref_focus))

def test_jones_adapter_angular_spectrum():
    """test jones adapter on propagation functions
    """

    from prysm.coordinates import make_xy_grid, cart_to_polar
    from prysm.geometry import circle
    from prysm.propagation import angular_spectrum
    from prysm.polynomials import hopkins
    from prysm.experimental.polarization import jones_adapter
    from prysm.conf import config

    N,M = 256,256
    wvl = 1e-6
    D = 2
    dx = N/D
    z = (D/2)**2 / (3*wvl) # Fresnel number of 3

    # set up a wave function for the on-diagonals
    x,y = make_xy_grid(N,diameter=D)
    r,t = cart_to_polar(x,y)
    rho = r/5
    phi = hopkins(0,4,0,rho,t,1)
    A = circle(1,r)

    wavefunction = A*np.exp(1j*2*np.pi/wvl*phi)

    # Set up jones data, numpy.ndarray of shape N,M,2,2
    # This is essentially a non-polarizing system
    jones = np.zeros([N,M,2,2],dtype=config.precision_complex) # this represents our "wavefunction"
    jones[...,0,0] = wavefunction
    jones[...,1,1] = wavefunction

    # test focus
    jones_prop = jones_adapter(jones,angular_spectrum,wvl,dx,z,Q=2)
    ref_prop = angular_spectrum(wavefunction,wvl,dx,z,Q=2)

    np.testing.assert_allclose((jones_prop[...,0,0],jones_prop[...,1,1]),(ref_prop,ref_prop))

def test_jones_adapter_methods():
    """test jones adapter on propagation methods of the Wavefront class
    """
    pass



