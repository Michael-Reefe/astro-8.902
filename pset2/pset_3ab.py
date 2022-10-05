import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}\usepackage{physics}"
plt.rcParams['font.family'] = "Times New Roman"

# using pi = G = r0 = 1
def Phi(rr0, alpha):
    # input: r/r0
    if alpha == 0:
        return 2/3 * rr0**2
    elif alpha == 1:
        return 2 * rr0
    elif alpha == 2:
        return 4 * np.log(rr0)
    elif alpha == 3:
        return -4 / rr0 * np.log(rr0)

def vrot(rr0, alpha):
    if alpha == 0:
        return rr0
    elif alpha == 1:
        return np.sqrt(rr0)
    elif alpha == 2:
        return np.ones(rr0.size)
    elif alpha == 3:
        return np.sqrt(1/rr0)

# Generate array of r/r0 scales
rr0 = np.linspace(0, 4, 100)

# Get potentials for different alphas
P1 = Phi(rr0, 1)
P2 = Phi(rr0, 2)
P3 = Phi(rr0, 3)

# Plot
fig, ax = plt.subplots()
ax.plot(rr0, P1, 'k-', label=r'$\alpha = 1$')    # Linear
ax.plot(rr0, P2, 'r-', label=r'$\alpha = 2$')    # Logarithmic
ax.plot(rr0, P3, 'b-', label=r'$\alpha = 3$')    # Log * 1/r
ax.set_xlabel(r'$r/r_0$')
ax.set_ylabel(r'$\Phi(r) / \pi G \rho_0 r_0^2$')
ax.set_ylim(-10, 10)
ax.legend()
plt.savefig('phi_plots.pdf', dpi=300, bbox_inches='tight')
plt.close()

# Get rotational velocities for different alphas
v1 = vrot(rr0, 1)
v2 = vrot(rr0, 2)
v3 = vrot(rr0, 3)

# Plot
fig, ax = plt.subplots()
ax.plot(rr0, v1, 'k-', label=r'$\alpha = 1$')   # r^1/2
ax.plot(rr0, v2, 'r-', label=r'$\alpha = 2$')   # Constant
ax.plot(rr0, v3, 'b-', label=r'$\alpha = 3$')   # r^-1/2
ax.set_xlabel(r'$r/r_0$')
ax.set_ylabel(r'$v_{\rm rot} / \sqrt{\frac{4}{3}\pi G \rho_0 r_0^2}$')
ax.legend()
plt.savefig('vrot_plots.pdf', dpi=300, bbox_inches='tight')
plt.close()
