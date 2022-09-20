from lib2to3.pgen2.token import AMPER
from re import X
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as opt, integrate as integ
from astropy.io import fits

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}\usepackage{physics}"
plt.rcParams['font.family'] = "Times New Roman"

# Function for reading in SDSS spectrum files
def read_spec(fits_file):

    # Load the data
    hdu = fits.open(fits_file)

    # Retrieve redshift from spectrum file (specobj table)
    specobj = hdu[2].data
    z = specobj['z'][0]

    t = hdu[1].data
    hdu.close()

    # Unpack the spectra
    galaxy = t['flux']
    wdisp = t['wdisp']

    # SDSS spectra are already log10-rebinned
    loglam_gal = t['loglam'] # This is the observed SDSS wavelength range, NOT the rest wavelength range of the galaxy
    lam_gal = 10**loglam_gal
    ivar = t['ivar'] # inverse variance
    noise = np.sqrt(1.0/ivar) # 1-sigma spectral noise
    and_mask = t['and_mask'] # bad pixels 
    bad_pix  = np.where(and_mask != 0)[0]

    ### Interpolating over bad pixels ############################

    # Get locations of nan or -inf pixels
    nan_gal   = np.where(~np.isfinite(galaxy))[0]
    nan_noise = np.where(~np.isfinite(noise))[0]
    inan = np.unique(np.concatenate([nan_gal,nan_noise]))

    # Interpolate over nans and infs if in galaxy or noise
    noise[inan] = np.nan
    noise[inan] = 1.0 if all(np.isnan(noise)) else np.nanmedian(noise)
    galaxy[inan] = np.nan
    galaxy[inan] = 1.0 if all(np.isnan(galaxy)) else np.nanmedian(galaxy)

    return lam_gal, galaxy, noise, wdisp, z

# Function for plotting spectra
def plot_spec(wave, flux, noise):
    fig, ax = plt.subplots(figsize=(16,6))
    ax.plot(wave, flux, 'k-')
    ax.fill_between(wave, flux-noise, flux+noise, color='k', alpha=0.5)
    ax.set_xlabel(r'$\lambda_{\rm obs}$ ($\AA$)')
    ax.set_ylabel(r'$F_\lambda$ ($10^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$)')
    ax.set_xlim(np.nanmin(wave), np.nanmax(wave))
    ax.set_title("SDSS Spectrum 0284-51943-0037")
    plt.savefig('0284-51943-0037_spectrum.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def plot_zoomed(wave, flux, noise, window, suffix):
    fig, ax = plt.subplots()
    ax.plot(wave[window], flux[window], 'k-')
    ax.fill_between(wave[window], (flux-noise)[window], (flux+noise)[window], color='k', alpha=0.5)
    ax.set_xlabel(r'$\lambda_{\rm obs}$ ($\AA$)')
    ax.set_ylabel(r'$F_\lambda$ ($10^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$)')
    ax.set_xlim(np.nanmin(wave[window]), np.nanmax(wave[window]))
    ax.set_ylim(np.nanmin(flux[window])-5, 2*np.nanmax(flux[window]))
    ax.set_title("SDSS Spectrum 0284-51943-0037")
    plt.savefig(f'0284-51943-0037_spectrum_zoomed_{suffix}.pdf', dpi=300, bbox_inches='tight')
    plt.close() 

wave, flux, noise, wdisp, z = read_spec("spec-0284-51943-0037.fits")
plot_spec(wave, flux, noise)

plot_zoomed(wave, flux, noise, np.where((wave > 7350) & (wave < 7500)), "ha")
plot_zoomed(wave, flux, noise, np.where((wave > 5400) & (wave < 5800)), "oiii")
plot_zoomed(wave, flux, noise, np.where((wave > 4100) & (wave < 4300)), "oii")

# FITTING LINES

def line_model(p, x):
    # Parameter vector: (constant, ampltiude, mean, dispersion)
    return p[0] + p[1] * np.exp( -(x-p[2])**2 / (2 * p[3]**2) )

# log of the likelihood
def lnlike(p, x, y, err):
    return -0.5 * np.nansum((y - line_model(p, x))**2 / err**2 + np.log(2*np.pi * err**2))

# log of the priors (upper/lower limits on parameters)
def lnprior(p, x):
    # Amplitude must be positive, mean must be within bounds, dispersion must be positive
    if (p[1] < 0) or (p[2] < min(x)) or (p[2] > max(x)) or (p[3] < 0) or (p[3] > 5):
        return -np.inf
    else:
        return 0.

# negative log of probability -- to be maximized
def neglnprob(p, x, y, err):
    return -lnlike(p, x, y, err)-lnprior(p, x)

# LINES TO BE FIT: Halpha, [N II], Hbeta, [O III], [O II]

# Rough estimate of the central wavelengths for each of these
w = lambda lo,hi: np.where((wave > lo) & (wave < hi))[0]
ha = wave[w(7415,7435)][np.nanargmax(flux[w(7415,7435)])]
nii_l = wave[w(7400,7415)][np.nanargmax(flux[w(7400,7415)])]
nii_r = wave[w(7440,7460)][np.nanargmax(flux[w(7440,7460)])]
hb = wave[w(5480,5520)][np.nanargmax(flux[w(5480,5520)])]
oiii_l = wave[w(5600,5620)][np.nanargmax(flux[w(5600,5620)])]
oiii_r = wave[w(5655,5670)][np.nanargmax(flux[w(5655,5670)])]
oii = wave[w(4200,4225)][np.nanargmax(flux[w(4200,4225)])]

# Get the velocity scale of the spectrum (the km/s scale for each pixel) -- we will need this later
frac = wave[1]/wave[0]    # constant wavelength fraction (spectrum is logarithmically binned)
assert np.isclose(frac, wave[-1]/wave[-2]), "Spectrum is not logarithmically binned!"

velscale = (frac**2 - 1)/(frac**2 + 1) * 299792.458    # velocity scale (km/s/pixel) from SR redshift eqn
flux_out = {}

for mu, label, suffix, rest in zip([ha, nii_l, nii_r, hb, oiii_l, oiii_r, oii], ["H$\\alpha$", "[N II] $\lambda$6548", "[N II] $\lambda$6583", "H$\\beta$", "[O III] $\lambda$4960", "[O III] $\lambda$5007", "[O II] $\lambda\lambda$3727,3729"],
    ["ha", "nii_l", "nii_r", "hb", "oiii_l", "oiii_r", "oii"], [6563, 6548, 6583, 4861, 4960, 5007, 3728]):

    # Window of +/- 10 angstroms
    window = np.where((wave > mu-15) & (wave < mu+15))[0]
    # Create pixel vector so that we can properly convert to a velocity scale
    pix = np.arange(len(wave[window]))
    # Normalize the spectrum before fitting
    norm = np.nanmedian(flux[np.where(((wave > 7360) & (wave < 7400)) | ((wave > 7460) & (wave < 7500)))])
    fluxnorm = flux[window] / norm
    fluxnorm_err = noise[window] / norm

    # Fit with scipy minimize
    popt = opt.minimize(neglnprob, x0=[1, 4, len(pix)/2, 3], args=(pix, fluxnorm, fluxnorm_err), method='Nelder-Mead')
    cont, amp, mu_pix, disp = popt.x
    # Convert parameters back into proper units
    # Flux units
    continuum = norm * cont
    amplitude = norm * amp
    # Wavelength units -- interpolate the wave vector with the final pixel value
    mean = np.interp(mu_pix, pix, wave[window])
    # Velocity dispersion in km/s
    dispersion = disp * velscale
    # Subtract the SDSS broadening in quadrature
    disp_res_i = np.interp(mu_pix, pix, wdisp[window]) * velscale
    dispersion = np.sqrt(dispersion**2 - disp_res_i**2)
    # Gaussian full-width at half-maximum
    fwhm = 2 * np.sqrt(2 * np.log(2)) * dispersion

    # Reconstruct the full model
    model = line_model([cont, amp, mu_pix, disp], pix) * norm
    # Integrated flux using the model WITHOUT the continuum
    flux_out[suffix] = integ.simps(line_model([0, amp, mu_pix, disp], pix) * norm * 1e-17, wave[window])

    # Measure redshifts!
    z_i = (mean - rest)/rest

    fig, ax = plt.subplots()
    ax.plot(wave[window], flux[window], 'k-', label='Data')
    ax.fill_between(wave[window], flux[window]-noise[window], flux[window]+noise[window], color='k', alpha=0.5)
    ax.plot(wave[window], model, 'r-', label='Model')
    ax.annotate("Continuum $ = $ %.2f \n Amplitude $ = $ %.2f \n Mean $ = $ %.0f $\\AA$ \n FWHM $ = $ %.0f km/s \n $\log_{10}(F) = $ %.2f \n $z = $ %.4f" % (continuum, amplitude, mean, fwhm, np.log10(flux_out[suffix]), z_i), (0.1, 0.85), xycoords='figure fraction', ha='left', va='top')
    ax.annotate(label, (mean, amplitude+continuum+1), ha='center', va='center')
    ax.set_xlim(np.nanmin(wave[window]), np.nanmax(wave[window]))
    ax.set_ylim(np.nanmin(flux[window])-1, amplitude+continuum+2)
    ax.legend(loc='upper right', frameon=False)
    ax.set_xlabel(r'$\lambda_{\rm obs}$ ($\AA$)')
    ax.set_ylabel(r'$F_\lambda$ ($10^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$)')
    ax.set_title("SDSS Spectrum 0284-51943-0037")
    plt.savefig(f"0284-51943-0037_model_{suffix}.pdf", dpi=300, bbox_inches='tight')
    plt.close()


# Line ratios
# We did not detect [O III], so use a 3-sigma upper limit on the flux instead
oiii_wave = np.argmin(np.sum((wave - 5007)**2))
# Flux threshold: integral of a guassian with amplitude equal to RMS of the flux and width equal to the instrumental dispersion
oiii_thresh = np.sqrt(2*np.pi) * np.std(flux[(wave > 4997) & (wave < 5017)]) * wdisp[oiii_wave] * (frac - 1) * 5007
oiii_3sig = 3 * oiii_thresh * 1e-17

o2_3 = np.log10(flux_out["oii"] / oiii_3sig)
print(f"log([O II]/[O III]) = {o2_3}")
n2_ha = np.log10(flux_out["nii_r"] / flux_out["ha"])
print(f"log([N II]/Ha) = {n2_ha}")
o3_hb = np.log10(oiii_3sig / flux_out["hb"])
print(f"log([O III]/Hb) = {o3_hb}")

# BPT plot
def blineplot(x, a, b, c):
    return (a / (x + b)) + c

def blinerplot(y, d, e):
    return ((d * y) + e)

bl = np.linspace(-1.27, -.01, num=100)
bl1 = np.linspace(-5., .35, num=100)
bl2 = np.linspace(-.18, .2, num=100)

Kauf = blineplot(bl, 0.61, -.05, 1.3)
Ke = blineplot(bl1, 0.61, -0.47, 1.19)

fig, ax = plt.subplots()
ax.plot(bl, Kauf, 'k-', label='Kauffmann+2003')
ax.plot(bl1, Ke, 'k--', label='Kewley+2001')
ax.plot(n2_ha, o3_hb, 'r.')
ax.arrow(n2_ha, o3_hb, dx=0, dy=-0.5, width=0.001, head_width=0.02, head_length=0.02, color='r')
ax.set_xlabel(r'$\log_{10}$([N II]/H$\alpha$)')
ax.set_ylabel(r'$\log_{10}$([O III]/H$\beta$)')
ax.set_xlim(-1.5, 1)
ax.set_ylim(-1.5, 1)
plt.savefig("0284-51943-0037_BPT.pdf", dpi=300, bbox_inches='tight')
plt.close()