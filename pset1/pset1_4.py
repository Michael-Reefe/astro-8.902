import numpy as np
from scipy import integrate

# Unitless luminosity density in terms of Lfrac = L/L*
def Phi_int(Lfrac, alpha):
    return Lfrac**(alpha+1) * np.exp(-Lfrac)

# Constants
Phi0 = 0.02            # Mpc^-3 h^-2
Lstar = 1e10           # Lsun

# Full integral over all L/L* space
L_tot = Phi0 * integrate.quad(Phi_int, 0, np.inf, args=(-1.1,))[0]
print(f'L_tot = {L_tot} L*')

# Nearby cluster, 4-mag limited regime
L_near = Phi0 * integrate.quad(Phi_int, 10**-1.6, np.inf, args=(-1.1,))[0]
print(f'L_near = {L_near} L*')
print(f'{L_near/L_tot * 100}% of L_tot')

# Far cluster, 0.5-mag limited regime
L_far = Phi0 * integrate.quad(Phi_int, 10**-0.2, np.inf, args=(-1.1,))[0]
print(f'L_far = {L_far} L*')
print(f'{L_far/L_tot*100}% of L_tot')
print(f'{100 - L_far/L_near*100}% of light lost relative to L_near')


######### luminosity density fractions below L* ############
for alpha in (-1, -1.25, -1.5, -1.85):
    rho = integrate.quad(Phi_int, 0, 1, args=(alpha,))[0] / integrate.quad(Phi_int, 0, np.inf, args=(alpha,))[0]
    print(f'For alpha = {alpha}: fraction below L* = {rho}')
