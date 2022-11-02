import numpy as np
from scipy.integrate import quad
from matplotlib import pyplot as plt

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = "Times New Roman"

# Constants
c = 299792.458   # Speed of light (km/s)

# Define cosmology: Flat LambdaCDM
H0 = 70                       # Hubble constant (t=0) (km/s/Mpc)
Om0 = 0.0                     # Omega_matter (t=0)
Or0 = 0.0                     # Omega_radiation (t=0)
Ode0 = 1.0                    # Omega_Lambda (t=0)
OK0 = 1 - Om0 - Or0 - Ode0    # Omega_K (t=0)

# Hubble distance scale in Mpc
DH = c/H0

# Dimensionless H(z) function
E = lambda z: np.sqrt(Om0*(1+z)**3 + Or0*(1+z)**4 + OK0*(1+z)**2 + Ode0)

# Comoving distance in Mpc given redshift z
def comoving_distance(z):
    return DH * quad(lambda z: 1/E(z), 0, z)[0]

# Proper distance in Mpc given redshift z
def proper_distance(z):
    if np.isclose(OK0, 0):
        return comoving_distance(z)
    elif OK0 > 0:
        return DH/np.sqrt(OK0) * np.sinh(np.sqrt(OK0) * comoving_distance(z)/DH)
    else:
        return DH/np.sqrt(np.abs(OK0)) * np.sin(np.sqrt(np.abs(OK0)) * comoving_distance(z)/DH)

# Angular diameter distance in Mpc given redshift z
def angular_diameter_distance(z):
    return proper_distance(z) / (1+z)

def luminosity_distance(z):
    return proper_distance(z) * (1+z)

# Choose physical size to be 1 Mpc
l = 1
# Evaluate angle for a range of redshifts
z = np.linspace(0, 10, 10000)
dA = np.array([angular_diameter_distance(zi) for zi in z])
# Get anglular sizes
theta = l / dA * 180/np.pi * 3600

# Get corresponding euclidean values
dL = np.array([luminosity_distance(zi) for zi in z])
theta_lum = l / dL * 180/np.pi * 3600

# Redshift where distance disagrees by 10% with theta_Euclid
diff = np.abs(dA - dL) / dA
p10 = np.where(diff > 0.1)[0]
if len(p10) > 0:
    z10 = z[p10][0]
else:
    z10 = np.nan

# Minimum angular size
pmin = np.argmin(theta)
zmin = z[pmin]

# Plot 1
fig, ax = plt.subplots()
region = np.where(z < 0.5)[0]
ax.plot(z[region], dA[region], 'r-', label='Angular Diameter Distance')
ax.plot(z[region], dL[region], 'k-', label='Luminosity Distance')
ax.axvline(z10, 0, 1, linestyle='--', color='k', alpha=0.5, label='10\\%% Deviation, $z = %.2f$' % z10)
ax.legend()
# ax.set_ylim(0, 1000)
ax.set_xlabel('$z$')
ax.set_ylabel('$d$ (Mpc)')
ax.set_title(r'$H_0 = %d$ km s$^{-1}$ Mpc$^{-1}$, $\Omega_m = %.1f$, $\Omega_\Lambda = %.1f$, $\Omega_K = %.1f$' % (H0, Om0, Ode0, OK0))
plt.savefig(f'distances_{H0}_{Om0}_{Ode0}.pdf', dpi=300, bbox_inches='tight')
plt.close()

# Plot 2
fig, ax = plt.subplots()
ax.plot(z, theta, 'r-')
# ax.plot(z, theta_lum, 'k-', label='Luminosity Distance')
ax.axvline(zmin, 0, 1, linestyle='--', color='k', alpha=0.5, label='Minimum, $z = %.2f$' % zmin)
ax.legend()
ax.set_ylim(0, 1000)
ax.set_xlabel('$z$')
ax.set_ylabel('$\\theta$ (arcsec)')
ax.set_title(r'$H_0 = %d$ km s$^{-1}$ Mpc$^{-1}$, $\Omega_m = %.1f$, $\Omega_\Lambda = %.1f$, $\Omega_K = %.1f$' % (H0, Om0, Ode0, OK0))
plt.savefig(f'angular_size_{H0}_{Om0}_{Ode0}.pdf', dpi=300, bbox_inches='tight')
plt.close()

# size at z=2 and z=10
z2 = l / angular_diameter_distance(2) * 180/np.pi * 3600
z10 = l / angular_diameter_distance(10) * 180/np.pi * 3600
print(f'Size at z=2: {z2}"')
print(f'Size at z=10: {z10}"')
