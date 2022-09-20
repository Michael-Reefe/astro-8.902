import numpy as np
import photutils
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.wcs import WCS
from scipy import optimize
from matplotlib import pyplot as plt

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}\usepackage{physics}"
plt.rcParams['font.family'] = "Times New Roman"

for filename, ra, dec, galname, radius in zip(["002126-4-0382", "003841-6-0121"], [176.454777448, 182.956389839], [5.338100283, 8.439073659], 
    ["Spiral Galaxy J114549.14+052017.1", "Elliptical Galaxy J121149.53+082620.6"], [20, 20]):

    mu_i = []
    for band in ('r', 'g'):

        hdur = fits.open(f"frame-{band}-{filename}.fits")

        # World Coordinate System (same for g and r)
        wcs = WCS(hdur[0].header)

        # Flux data
        flux_r = hdur[0].data

        gal_coords = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='fk5')
        gal_coords = wcs.world_to_pixel(gal_coords)

        # BUNIT = 1 nanomaggy = 3.631e-6 Jy
        # Convert nanomaggies to erg/s/cm2/Hz
        flux_r = flux_r * 3.631e-6 * 1e-23

        # Find the peak brightness of the galaxy
        width = 100
        flux_r_small = flux_r[int(gal_coords[1])-width:int(gal_coords[1])+width, int(gal_coords[0])-width:int(gal_coords[0])+width]
        peak_r = np.argwhere(flux_r_small == np.nanmax(flux_r_small))[0] + (int(gal_coords[1])-width, int(gal_coords[0])-width) 
        # Convert to RA/Dec
        # peak_r = wcs.pixel_to_world(peak_r[1], peak_r[0])
        peak_r = wcs.pixel_to_world(*gal_coords)

        # Find the median sky brightness using an annulus around the galaxy
        annulus_r = photutils.aperture.SkyCircularAnnulus(peak_r, r_in=30*u.arcsec, r_out=40*u.arcsec)
        annulus_pix_r = annulus_r.to_pixel(wcs)
        sky_flux_r = photutils.aperture.ApertureStats(flux_r, annulus_pix_r).median

        # Subtract sky flux
        flux_r -= sky_flux_r

        # radii of concentric circular apertures (arcsec)
        # plate scale of SDSS is ~0.4 arcsec/px
        r = np.arange(0.4, radius + 0.2, 0.2)

        # 1D surface brightness profiles
        sb_r_1d = np.zeros(r.size)

        # Create central circular apertures with increasing radii
        fr_prev = 0.
        for i in range(len(r)):
            if i == 0:
                ap_r = photutils.aperture.SkyCircularAperture(peak_r, r=r[i]*u.arcsec)
            else:
                ap_r = photutils.aperture.SkyCircularAnnulus(peak_r, r_in=r[i-1]*u.arcsec, r_out=r[i]*u.arcsec)
            ap_r_pix = ap_r.to_pixel(wcs)
            # Measure flux in each aperture
            fi_r = photutils.aperture_photometry(flux_r, ap_r_pix)['aperture_sum'][0]
            # Divide by solid angle to convert to surface brightness (erg / s / cm^2 / Hz / arcsec^2)
            # dΩ = 2πsinθdθ
            omega = np.sin(r[i]/3600*np.pi/180) * 0.2 * (360 * 3600)
            sb_r_1d[i] = fi_r/omega

        # Convert to magnitude
        mu = lambda I: -2.5 * np.log10(I) - 48.6
        mu_r_1d = mu(sb_r_1d)

        ### FITTING ###

        # Sersic profile
        bn = lambda n: 2*n - 1/3 + 4/(405*n) + 46/(25515*n**2) + 131/(1148175*n**3) - 2194697/(30690717750*n**4)
        def sersic(p, r):
            return p[0] * np.exp(-bn(p[2]) * ((r/p[1])**(1/p[2]) - 1))

        # difference of squares in log space
        def logsquares(p, r, I, func):
            return 0.5 * np.sum((np.log(func(p, r)) - np.log(I))**2)

        # fit least squares with scipy
        popt_1 = optimize.minimize(logsquares, x0=[sb_r_1d[0], 1, 1], args=(r, sb_r_1d, sersic), method='Nelder-Mead')
        # fitted model parameters
        Ie, re, n = popt_1.x
        # surface brightness models
        model_1 = mu(sersic([Ie, re, n], r))

        print(galname)

        # Plot surface brightness profiles in the r band
        fig, ax = plt.subplots()
        ax.plot(r, mu_r_1d, 'k-', lw=2, label=f'$%s$ data' % band)
        ax.plot(r, model_1, f'{band}--', lw=2, label=r'$I_e\exp\{-b_n[(r/r_e)^{1/n}-1]\}$')
        ax.legend(frameon=False, loc='lower left', fontsize=14)
        ax.set_xlabel(r'$r$ (arcsec)', fontsize=16)
        ax.set_ylabel(r'$\mu$ (mag arcsec$^{-2}$)', fontsize=16)
        ax.set_xticks([0, 5, 10, 15, 20])
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        ax.annotate("$\\mu_e = $%.2f \n $r_e = \\ $%.2f \n $n = \\ $%.2f" % (mu(Ie), re, n), (16,22), ha='center', va='center', fontsize=14)
        ax.invert_yaxis()
        # ax.set_yscale('log')
        ax.set_title(galname, fontsize=16)
        fig.savefig(f'{filename}_{band}_1.pdf', dpi=300, bbox_inches='tight')
        plt.close()

        resid = np.sum((mu_r_1d - model_1)**2)
        print(f'Sersic Residuals: {resid}')

        # Exponential (n=1) profile
        def sersicexp(p, r):
            return p[0] * np.exp(-bn(p[2]) * ((r/p[1])**(1/p[2]) - 1)) + p[3] * np.exp(-r/p[4])

        popt_2 = optimize.minimize(logsquares, x0=[sb_r_1d[0], 5, 0.5, sb_r_1d[0], 1/0.15], args=(r, sb_r_1d, sersicexp), method='Nelder-Mead')
        
        Ie, re, n, I0, h = popt_2.x
        model_2 = mu(sersicexp(popt_2.x, r))

        # Plot surface brightness profiles in the r band
        fig, ax = plt.subplots()
        ax.plot(r, mu_r_1d, 'k-', lw=2, label=f'$%s$ data' % band)
        ax.plot(r, model_2, f'{band}--', lw=2, label=r'$I_e\exp\{-b_n[(r/r_e)^{1/n}-1]\} + I_0\exp\{-r/h\}$')
        ax.legend(frameon=False, loc='lower left', fontsize=14)
        ax.set_xlabel(r'$r$ (arcsec)', fontsize=16)
        ax.set_ylabel(r'$\mu$ (mag arcsec$^{-2}$)', fontsize=16)
        ax.set_xticks([0, 5, 10, 15, 20])
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        # ax.set_ylim(19, 26)
        ax.annotate("$\\mu_e = $%.2f \n $r_e = \\ $%.2f \n $n = \\ $%.2f \n $\\mu_0 = \\ $%.2f \n $h = \\ $%.2f" % (mu(Ie), re, n, mu(I0), h), (16,22), ha='center', va='center', fontsize=14)
        ax.invert_yaxis()
        # ax.set_yscale('log')
        ax.set_title(galname, fontsize=16)
        fig.savefig(f'{filename}_{band}_2.pdf', dpi=300, bbox_inches='tight')
        plt.close()

        resid = np.sum((mu_r_1d - model_2)**2)
        print(f'Sersic + Exp Residuals: {resid}')

        mu_i.append(mu_r_1d)

    
    # RELATIVE surface brightness: g-r
    r = np.arange(0.4, 20 + 0.2, 0.2)
    mu_rel = mu_i[1] - mu_i[0]

    fig, ax = plt.subplots()
    ax.plot(r, mu_rel, 'k-')
    ax.set_xlabel(r'$r$ (arcsec)', fontsize=16)
    ax.set_ylabel(r'$\mu_g-\mu_r$ (mag arcsec$^{-2}$)', fontsize=16)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    # ax.invert_yaxis()
    ax.set_title(galname, fontsize=16)
    fig.savefig(f'{filename}_g-r.pdf', dpi=300, bbox_inches='tight')
    plt.close()