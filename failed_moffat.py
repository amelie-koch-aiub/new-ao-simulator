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
from hcipy import *
import math


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

focal_length = 5.6
platescale = 206265/focal_length
sci_max = 5500e-10
telescope_diameter = 0.8
sci_cam_px_size = 2.9e-6
fov_sci = 2
spatial_res = 1.22 * sci_max * focal_length / telescope_diameter # all variables use meters
num_airy = fov_sci/platescale / spatial_res
q = math.ceil(spatial_res / sci_cam_px_size) # q = number of pixels between rings, num_airy = number of rings that should be visible
focal_grid_sci = make_focal_grid(q=q, num_airy=num_airy, spatial_resolution=spatial_res)

data = np.load('simulations/sim_82/moffat_failed_data.npy')

x = focal_grid_sci.x.reshape([46, 46])[0, :]
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
print('fwhm of data without fit: ', fwhm_no_fit)


x0 = x[np.argmax(data)]
background = np.median(data)
amplitude = data.max() - background
sigma = 0.5 # in arcsec
alpha = 2.5
p0 = [amplitude, x0, sigma, background]
bounds = ([0, np.min(x), 0, -np.inf],
            [np.inf, np.max(x), np.max(x), np.inf])

popt, pcov = curve_fit(gaussian_model, x, data, p0=p0, bounds=bounds, max_nfev=1000000) #p0=p0, 
fit = gaussian_model(x, *popt)
amplitude_fit, x0_fit, sigma_fit, background_fit = popt
#fwhm = 2 * gamma_fit * np.sqrt(2**(1 / alpha_fit) - 1) # in arcsec
fwhm = 2.354 * sigma_fit


#print(f'{amplitude_fit=}, {x0_fit=}, {gamma_fit=}, {alpha_fit=}, {background_fit=}')
print(f'{fwhm=}', popt)

angular_fwhm = fwhm # in arcsec
angular_x = x # in arcsec

plt.clf()
plt.plot(angular_x, data, label = f' slice', drawstyle='steps-mid')
plt.plot(angular_x, fit, label=f'Moffat: {x0_fit=}, {sigma_fit=}')
plt.axvline(angular_fwhm/2+angular_x[np.argmax(data)], linestyle='dashed', color='red', label='FWHM radius')
plt.axvline(-angular_fwhm/2+angular_x[np.argmax(data)], linestyle='dashed', color='red')
plt.xlim(-2, 2)
plt.ylim(3e-6, 1)
plt.xlabel('Angular separation [arcsec]')
plt.ylabel('Normalized Intensity')
plt.legend()
plt.show()