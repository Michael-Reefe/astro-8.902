import numpy as np
from astropy.convolution import convolve, Gaussian1DKernel
from scipy import integrate
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}\usepackage{physics}"
plt.rcParams['font.family'] = "Times New Roman"

# Exponential intensity function
def I(I0, r, h):
    return I0 * np.exp(-r/h)
# Normal distribution
def normal(r, r0, sigma):
    return 1/np.sqrt(2*np.pi*sigma**2) * np.exp(-(r-r0)**2 / (2 * sigma**2))

# Array of points r/h (h=1)
rh = np.linspace(0, 10, 1000)
# Convolve with Gaussian (only the edges are affected)
convI = convolve(I(1, rh, 1), kernel=Gaussian1DKernel(10))

fig, ax = plt.subplots()
ax.plot(rh, (I(1, rh, 1)), 'k-', label='Exponential')
ax.plot(rh, (convI), 'r-', label='Convolved')
ax.legend()
ax.set_xlabel(r'$r/h$')
ax.set_ylabel(r'$I$')
plt.savefig('gaussian_convolution.pdf', dpi=300, bbox_inches='tight')
plt.close()

# Now do it with a Sersic function
bn = lambda n: 2*n - 1/3 + 4/(405*n) + 46/(25515*n**2) + 131/(1148175*n**3) - 2194697/(30690717750*n**4)
def sersic(Ie, n, re, r):
    return Ie * np.exp(-bn(n) * ((r/re)**(1/n) - 1))

convS = convolve(sersic(1, 4, 1, rh), kernel=Gaussian1DKernel(10))

fig, ax = plt.subplots()
ax.plot(rh, np.log10(sersic(1, 4, 1, rh)), 'k-', label='Sersic ($n=4$)')
ax.plot(rh, np.log10(convS), 'r-', label='Convolved')
ax.legend()
ax.set_xlabel(r'$r/r_e$')
ax.set_ylabel(r'$\log_{10} I$')
plt.savefig('sersic_convolution.pdf', dpi=300, bbox_inches='tight')
plt.close()