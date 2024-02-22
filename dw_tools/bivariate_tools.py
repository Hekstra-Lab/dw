from scipy.stats import multivariate_normal, norm, vonmises, multivariate_t
from scipy.special import i0 as I0
import numpy as np


def Norm2D_polar(phi1, phi2, R1, R2, mean_xy, cov_xy):
    '''
    Returns the value of the bivariate complex normal distribution in polar coordinates.

        Parameters:
            phi1 (float): phase of the first complex rv
            phi2 (float): phase of the second complex rv
            R1 (float):   amplitude of the first complex rv
            R2 (float):   amplitude of the second complex rv
            mean_xy ([4,1] numpy array): mean ([real(rv1), imag(rv1), real(rv2), imag(rv2)])
            cov_xy ([4,4] np array): corresponding covariance matrix

        Returns:
            value of the PDF in polar coordinates (taking into account the jacobian)
    '''
    x = [R1 * np.cos(phi1), R1 * np.sin(phi1), R2 * np.cos(phi2), R2 * np.sin(phi2)]
    return R1 * R2 * multivariate_normal.pdf(x, mean_xy, cov_xy)


def Bivariate_Rice_f(theta, R1, R2, K, p1, v):
    '''
    Integrand for the Bivariate_Rice function.
    '''
    #   p1 = (1+K)/beta # beta only every appears as (1+K)/beta
    p2 = 1 - v**2
    prefactor_1 = p1**2 * R1 * R2 / (2 * np.pi * p2)
    pref2_int_1 = np.exp(
        -2 * K / (1 + v) - (p1 * (R1**2 + R2**2)) / (2 * p2) + v * p1 * R1 * R2 * np.cos(theta) / p2
    )  # this is more stable
    integrand_2 = I0(
        np.sqrt((2 * K * p1 * (R1**2 + R2**2 + 2 * R1 * R2 * np.cos(theta)))) / (1 + v)
    )
    return prefactor_1 * pref2_int_1 * integrand_2


def Bivariate_Rice_wrapper(E1, E2, rx, E3=1.0, r=0.0, Sigma=1.0):
    '''
    PDF of the bivariate Rice distribution following eq. 15 of Abu-Dayya and Beaulieu.
    Wraps around Bivariate_Rice(R1, R2, K, p1, v)
    Parameters:
        E1 (float):  value of the first amplitude
        E2 (float):  value of the second amplitude
        rx (float):  double-wilson parameter for correlation between ON and OFF structure factors components
        E3 (float):  value of the reference amplitude, if any (default: 1).
        r (float):   double-wilson parameter correlation of E1 or E2 components with the reference data set (default: 0).
        Sigma (float): the scale factor for intensities (default: 1),

    Returns:
        a float corresponding to the value of the PDF of the bivariate Rice distribution
    '''
    R1 = E1
    R2 = E2
    R3 = r * E3
    K = R3**2 / (Sigma * (1 - r**2))
    p1 = 2 / (Sigma * (1 - r**2))
    v = (rx - r**2) / (1 - r**2)
    return Bivariate_Rice(R1, R2, K, p1, v)


def Bivariate_Rice(R1, R2, K, p1, v):
    '''
    PDF of the bivariate Rice distribution following eq. 15 of Abu-Dayya and Beaulieu
    Parameters:
        R1 (float): value of the first amplitude
        R2 (float): value of the second amplitude
        K (float):  equal to R3**2/(Sigma*(1-r**2)), with
            R3 the amplitude on which we condition, Sigma the scale factor for intensities,
            and r the correlation between R1 or R2 and R3.
        p1 (float): equal to (1+K)/beta = 2/(Sigma*(1-r**2))
        v (float): equal to (rx - r**2)/(1-r**2)

    Returns:
        a float corresponding to the value of the PDF of the bivariate Rice distribution
    '''
    # Setting up the integration:
    deg = 100  # Degree of underlying polynomial approximation
    grid, weights = np.polynomial.chebyshev.chebgauss(deg)

    a = 0.0
    b = 2.0 * np.pi

    # Change of interval
    theta = (b - a) * grid / 2.0 + (a + b) / 2.0
    prefactor = (b - a) / 2.0

    # Reweight for general functional form
    # Each flavor of gauss quadrature has a different formula for this
    w = weights * np.sqrt(1 - grid**2.0)
    # print(Bivariate_Rice_f(theta, R1, R2, K, p1, v).shape)
    return prefactor * w @ Bivariate_Rice_f(theta, R1, R2, K, p1, v)


def FoldedNorm2D_wrapper(E1, E2, rx, E3=1.0, r=0.0, Sigma=1.0):
    '''
    PDF of the bivariate Folded Normal distribution.
    Wraps around FoldedNorm2D(R1, R2, mean, cov)
    Parameters:
        E1 (float):  value of the first amplitude
        E2 (float):  value of the second amplitude
        rx (float):  double-wilson parameter for correlation between ON and OFF structure factors components
        E3 (float):  value of the reference amplitude, if any (default: 1).
        r (float):   double-wilson parameter correlation of E1 or E2 components with the reference data set (default: 0).
        Sigma (float): the scale factor for intensities (default: 1),

    Returns:
        a float corresponding to the value of the PDF of the bivariate FoldedNormal distribution
    '''
    mean = r * np.abs(E3) * np.asarray([1, 1])  # just in case
    cov = Sigma * np.asarray([[1 - r**2, rx - r**2], [rx - r**2, 1 - r**2]])
    return FoldedNorm2D(E1, E2, mean, cov)


def FoldedNorm2D_vect_wrapper(E1, E2, rx, E3=1.0, r=0.0, Sigma=1.0):
    '''
    PDF of the bivariate Folded Normal distribution.
    Wraps around FoldedNorm2D(R1, R2, mean, cov)
    Parameters:
        E1 (float):  (N,) values of the first amplitude
        E2 (float):  (N,) values of the second amplitude
        rx (float):  double-wilson parameter for correlation between ON and OFF structure factors components
        E3 (float):  value(s) of the reference amplitude, if any (default: 1).
        r (float):   double-wilson parameter correlation of E1 or E2 components with the reference data set (default: 0).
        Sigma (float): the scale factor for intensities (default: 1),

    Returns:
        a float corresponding to the value of the PDF of the bivariate FoldedNormal distribution
    '''
    mean = np.array([E3, E3])
    cov = Sigma * np.asarray([[1 - r**2, rx - r**2], [rx - r**2, 1 - r**2]])
    return FoldedNorm2D_all(E1, E2, mean, cov)


def FoldedNorm2D(R1, R2, mean, cov):
    pp = multivariate_normal.pdf(np.array([R1, R2]), mean=mean, cov=cov, allow_singular=False)
    pm = multivariate_normal.pdf(np.array([R1, -R2]), mean=mean, cov=cov, allow_singular=False)
    mp = multivariate_normal.pdf(np.array([-R1, R2]), mean=mean, cov=cov, allow_singular=False)
    mm = multivariate_normal.pdf(np.array([-R1, -R2]), mean=mean, cov=cov, allow_singular=False)
    return pp + pm + mp + mm


def FoldedNorm2D_all(R1, R2, mean, cov):
    pp = multivariate_normal.pdf(
        np.concatenate((R1, R2), axis=1), mean=mean, cov=cov, allow_singular=False
    )
    pm = multivariate_normal.pdf(
        np.concatenate((R1, -R2), axis=1), mean=mean, cov=cov, allow_singular=False
    )
    mp = multivariate_normal.pdf(
        np.concatenate((-R1, R2), axis=1), mean=mean, cov=cov, allow_singular=False
    )
    mm = multivariate_normal.pdf(
        np.concatenate((-R1, -R2), axis=1), mean=mean, cov=cov, allow_singular=False
    )
    return pp + pm + mp + mm


def BTFN(angles, cov, mu=0, nu=0):
    '''
    Bivariate torus folded normal
    Parameters:
        angles: array-like; where the BTFN needs to be evaluated
        cov: covariance matrix
        mu: mean phase 1 (0 by default, as often dictated by symmetry)
        nu: mean phase 2 (0 by default)
    Returns:
        PDF of the BTFN
    '''
    kmax = np.ceil(3 * np.sqrt(cov[0, 0]) / (2 * np.pi)).astype('int')
    lmax = np.ceil(3 * np.sqrt(cov[1, 1]) / (2 * np.pi)).astype('int')
    L = angles.shape[0]
    mean = np.array([mu, nu])
    tmp = np.array(
        [
            multivariate_normal.pdf(
                angles
                + np.repeat(np.array([k * 2 * np.pi, l * 2 * np.pi]).reshape(1, -1), L, axis=0),
                mean,
                cov,
            )
            for k in range(-kmax, kmax + 1)
            for l in range(-lmax, lmax + 1)
        ]
    )
    return np.sum(tmp, axis=0)


def BVTFST(angles, cov, mu=0, nu=0, df=8):
    '''
    Bivariate torus folded Student t PDF
    Parameters:
        angles: array-like; where the BTFN needs to be evaluated
        cov: covariance matrix
        mu: mean phase 1 (0 by default, as often dictated by symmetry)
        nu: mean phase 2 (0 by default)
    Returns:
        PDF of the BTFN
    '''
    kmax = np.ceil(4 * np.sqrt(cov[0, 0]) / (2 * np.pi)).astype('int')
    lmax = np.ceil(4 * np.sqrt(cov[1, 1]) / (2 * np.pi)).astype('int')
    L = angles.shape[0]
    mean = np.array([mu, nu])
    tmp = np.array(
        [
            multivariate_t.pdf(
                angles
                + np.repeat(np.array([k * 2 * np.pi, l * 2 * np.pi]).reshape(1, -1), L, axis=0),
                mean,
                cov,
                df,
            )
            for k in range(-kmax, kmax + 1)
            for l in range(-lmax, lmax + 1)
        ]
    )
    return np.sum(tmp, axis=0)
