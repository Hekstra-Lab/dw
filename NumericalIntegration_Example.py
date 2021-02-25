import numpy as np

deg = 100 #Degree of underlying polynomial approximation
grid, weights = np.polynomial.chebyshev.chebgauss(deg)

def f(p):
    return np.sin(x)**2.

a = 0.
b = 2.*np.pi

#Change of interval
x = (b-a)*grid/2. + (a+b)/2.
prefactor = (b-a)/2.

#Reweight for general functional form
#Each flavor of gauss quadrature has a different
#formula for this
w = weights*np.sqrt(1-grid**2.)


print(f"int of f(x) from {a} to {b}  = {prefactor*w@f(x)}")

