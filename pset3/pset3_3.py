import numpy as np
from scipy.integrate import quad

H0_Gyr = 2.26856e-18 * (365.25 * 24 * 3600 * 1e9)  # H0 in 1/Gyr
Om0 = 0.3
Ode0 = 0.7
Or0 = 1e-5
OK0 = 1 - Om0 - Ode0 - Or0
print(f'1/H0 = {1/H0_Gyr} Gyr')

def age(z):
    # Calculate the age of the universe in Gyr for a given redshift using Friedmann equations
    return 1/H0_Gyr * quad(lambda zi: 1/np.sqrt(Om0*(1+zi)**5 + Or0*(1+zi)**6 + OK0*(1+zi)**4 + Ode0*(1+zi)**2), 
        z, np.inf, epsabs=1e-14)[0]

# Get age for various redshifts
print(f't(z = 0)   = {age(0.)} Gyr')
print(f't(z = 0.7) = {age(0.7)} Gyr')
print(f't(z = 3)   = {age(3)} Gyr')
print(f't(z = 10)  = {age(10)} Gyr')
