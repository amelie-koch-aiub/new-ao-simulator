import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy import ndimage
import astropy.io.fits as fits
import numpy as np
from tqdm import tqdm
from scipy import ndimage
from skimage import measure
import os
from datetime import datetime
from astropy.time import Time
from scipy.integrate import cumulative_trapezoid
from astropy.timeseries import LombScargle
from scipy.interpolate import interp1d
from photutils.psf import fit_fwhm
from numpy import unravel_index
from photutils.psf import CircularGaussianPRF, fit_2dgaussian

from scipy.optimize import curve_fit
from astropy.modeling.functional_models import Moffat1D
import numpy as np
import scipy.ndimage as scind


# def analyze_airy_psf(image,
#                      search_radius=None,
#                      smooth_sigma=1.0,
#                      fit_radius=4,
#                      dr=0.25):

#     center = image
#     x0, y0 = unravel_index(center.argmax(), center.shape) 

#     tabl = fit_2dgaussian(image, fix_fwhm=False, fit_shape=7).results
#     #x0 = tabl["x_fit"][0]
#     #y0 = tabl["y_fit"][0]

#     fwhm = tabl["fwhm_fit"][0]
#     fwhm_err = tabl["fwhm_err"][0]

#     return {
#         "x0": x0,
#         "y0": y0,
#         "fwhm": fwhm,
#         }

# def mfgs(image: np.ndarray, kernel_size:int=3) -> float:
#     """
#     Median filter gradiant similarity from Deng et. al., 2015.
#     Used for tracking quality of image
#     """
#     med_img = scind.median_filter(image, size=kernel_size)
#     grad_img = np.gradient(image)

#     grad = (np.sum(np.abs(grad_img[0][image != 0])) + np.sum(np.abs(grad_img[1][image != 0])))
#     med_grad = np.sum(np.abs(np.gradient(med_img)))

#     mfgs_metric = ((2 * med_grad * grad) /
#                    (med_grad**2 + grad**2))

#     return mfgs_metric

# import numpy as np
# import scipy.ndimage as scind


def mfgs(psf, kernel_size=3):
    """
    Median-filtered gradient similarity (MFGS).

    Parameters
    ----------
    psf : HCIPy Wavefront or ndarray
        HCIPy PSF Wavefront, or a 2D intensity array.
    kernel_size : int
        Size of the median-filter kernel.

    Returns
    -------
    float
        MFGS metric, ranging from approximately 0 to 1.
        Higher values indicate greater similarity between the
        original and median-filtered gradient fields.
    """

    # If an HCIPy Wavefront is supplied, use its intensity.
    if hasattr(psf, "intensity"):
        image = np.asarray(psf.intensity).reshape(psf.grid.shape)
    else:
        image = np.asarray(psf).reshape(psf.grid.shape)

    # Make sure we're dealing with a 2D image.
    if image.ndim != 2:
        raise ValueError(
            f"Expected a 2D PSF, got shape {image.shape}"
        )

    # Median-filtered image
    med_img = scind.median_filter(
        image,
        size=kernel_size
    )

    # Gradients
    grad_y, grad_x = np.gradient(image)
    med_grad_y, med_grad_x = np.gradient(med_img)

    # Gradient magnitudes
    grad_mag = np.sqrt(
        grad_x**2 + grad_y**2
    )

    med_grad_mag = np.sqrt(
        med_grad_x**2 + med_grad_y**2
    )

    # Optional: ignore pixels where the PSF is essentially zero.
    mask = image > 0

    grad = np.sum(grad_mag[mask])
    med_grad = np.sum(med_grad_mag[mask])

    # Avoid division by zero.
    denominator = med_grad**2 + grad**2

    if denominator == 0:
        return 0.0

    return 2 * med_grad * grad / denominator

def find_nearest(array, value):
    """Finds index of closest value in array"""
    return np.abs(array-value).argmin()

def moffat_model(x, amplitude, x0, gamma, alpha, background):
    return (
        Moffat1D.evaluate(
            x,
            amplitude,
            x0,
            gamma,
            alpha
        )
        + background
    )

def gaussian_model(x, amplitude, x0, sigma, background):
    return background + amplitude * np.exp(-(x - x0)**2 / (2 * sigma**2))

def find_gaussian_fwhm(psf, psf_uncorr, platescale, simulation_id, timestep, axis='x', plot = True, fails = 0):
    psf_shape = psf.grid.shape
    focal_grid_sci = psf.grid # ? check this.

    if axis == 'x':
        image_slice = psf.shaped[:, psf_shape[0] // 2]
        uncorr_slice = psf_uncorr.shaped[:, psf_shape[0] // 2]
    else:
        image_slice = psf.shaped[psf_shape[0] // 2, :]
        uncorr_slice = psf_uncorr.shaped[psf_shape[0] // 2, :]

    data = image_slice / psf.max()
    data_uncorr = uncorr_slice / psf_uncorr.max()

    x = focal_grid_sci.x.reshape(psf_shape)[0, :] # in meters
    x *= platescale
    x0 = x[np.argmax(data)]
    background = np.median(data)
    amplitude = data.max() - background
    sigma = 0.5 # in arcsec
    p0 = [amplitude, x0, sigma, background]
    bounds = ([0, np.min(x), 0, -np.inf],
                [np.inf, np.max(x), np.max(x), np.inf])

    popt, pcov = curve_fit(gaussian_model, x, data, p0=p0, bounds=bounds, max_nfev=10000) 
    fit = gaussian_model(x, *popt)
    amplitude_fit, x0_fit, sigma_fit, background_fit = popt
    fwhm = 2.354 * sigma_fit

    if plot:
        plt.clf()
        plt.plot(x, data, label = f'{axis} slice')
        plt.plot(x, data_uncorr, label = f'uncorrected {axis} slice')
        plt.plot(x, fit, label=f'x0={round(x0_fit,4)}, $\sigma$={round(sigma_fit,4)}')
        plt.axvline(fwhm/2+x0_fit, linestyle='dashed', color='red', label='FWHM radius')
        plt.axvline(-fwhm/2+x0_fit, linestyle='dashed', color='red')
        plt.ylim(3e-6, 1)
        plt.xlabel('Angular separation [arcsec]')
        plt.ylabel('Normalized Intensity')
        plt.legend()
        plt.savefig(f'simulations/sim_{simulation_id}/psf_profile_{axis}_iter{timestep}.png')
        plt.clf()
    
    return fwhm



    

    
        
    


def find_moffat_fwhm(psf, psf_uncorr, platescale, simulation_id, timestep, axis='x', plot = True, fails = 0):
    psf_shape = psf.grid.shape
    print(psf_shape)
    focal_grid_sci = psf.grid # ? check this.

    if axis == 'x':
        image_slice = psf.shaped[:, psf_shape[0] // 2]
        uncorr_slice = psf_uncorr.shaped[:, psf_shape[0] // 2]
    else:
        image_slice = psf.shaped[psf_shape[0] // 2, :]
        uncorr_slice = psf_uncorr.shaped[psf_shape[0] // 2, :]

    data = image_slice / psf.max()
    data_uncorr = uncorr_slice / psf_uncorr.max()

    background = np.median(data)
    amplitude = data.max() - background
    x = focal_grid_sci.x.reshape(psf_shape)[0, :] # in meters
    # x0 = x[np.argmax(data)]
    # gamma = 0.5e-6 # in meters
    # alpha = 2.5
    # p0 = [amplitude, x0, gamma, alpha, background]
    # bounds = ([0, np.min(x), 0, 1, -np.inf],
    #             [np.inf, np.max(x), np.max(x), 10, np.inf])

    # # Fit
    # try:
    #     popt, pcov = curve_fit(moffat_model, x, data, bounds=bounds, max_nfev=10000) #p0=p0, 
    #     fit = moffat_model(x, *popt)
    #     amplitude_fit, x0_fit, gamma_fit, alpha_fit, background_fit = popt
    #     fwhm = 2 * gamma_fit * np.sqrt(2**(1 / alpha_fit) - 1) # in meters

    #     angular_fwhm = fwhm*platescale # in arcsec
    #     angular_x = x*platescale # in arcsec

    x *= platescale

    crossings = np.where(np.diff(np.sign(data - 0.5)) != 0)
    i1 = crossings[0][0]
    i2 = crossings[0][1]

    half_max_x1 = (
        x[i1]
        + (0.5 - data[i1])
        * (x[i1 + 1] - x[i1])
        / (data[i1 + 1] - data[i1])
    )

    half_max_x2 = (
        x[i2]
        + (0.5 - data[i2])
        * (x[i2 + 1] - x[i2])
        / (data[i2 + 1] - data[i2])
    )

    fwhm_no_fit = half_max_x2 - half_max_x1 

    angular_x = x

    if plot:
        plt.clf()
        plt.plot(angular_x, data, label = f'{axis} slice')
        plt.plot(angular_x, data_uncorr, label = f'uncorrected {axis} slice')
        plt.plot(angular_x, fit, label=f'Moffat: {x0_fit=}, {gamma_fit=}')
        # plt.axvline(angular_fwhm/2+angular_x[np.argmax(data)], linestyle='dashed', color='red', label='FWHM radius')
        # plt.axvline(-angular_fwhm/2+angular_x[np.argmax(data)], linestyle='dashed', color='red')
        plt.axvline(half_max_x1, linestyle='dashed', color='red', label='FWHM radius')
        plt.axvline(half_max_x2, linestyle='dashed', color='red')
        #plt.xlim(x[0], x[-1])
        plt.ylim(3e-6, 1)
        plt.xlabel('Angular separation [arcsec]')
        plt.ylabel('Normalized Intensity')
        plt.legend()
        plt.savefig(f'simulations/sim_{simulation_id}/psf_profile_{axis}_iter{timestep}.png')
        plt.clf()
    
    return angular_fwhm

    # except RuntimeError:
    #     if fails == 0:
    #         print(f'{p0=}')
    #         np.save(f"simulations/sim_{simulation_id}/moffat_failed_data.npy", data)
    #     # plt.clf()
    #     # plt.plot(x*platescale, data, label = f'{axis} slice')
    #     # plt.savefig(f'simulations/sim_{simulation_id}/psf_profile_{axis}_problem_{timestep}.png')
    #     # plt.clf()

    #     return np.inf


    


    
