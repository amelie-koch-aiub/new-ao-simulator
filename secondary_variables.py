import yaml 
import math
from pathlib import Path
import astropy.io.fits as fits
import numpy as np
from hcipy import *
from transmitted_power import transmitted_power
from PIL import Image
import matplotlib.pyplot as plt


def get_secondary_variables(simulation_id):
    config_file = f'configs/config_{simulation_id}.yaml'
    with open(config_file) as f:
            globals().update(yaml.safe_load(f))

        # set some telescope parameters
    platescale = 206265 / focal_length # [arcsec / m]
    central_obscuration_ratio = central_obscuration / telescope_diameter
    area_aperture = math.pi*(telescope_diameter/2)**2 - math.pi*(central_obscuration/2)**2 - 0.5*(telescope_diameter-central_obscuration)*spider_width*num_spiders # m^2

    # set some WFS parameters
    px_per_subap = round(fov_wfs / px_scale, 0)
    num_pupil_pixels = px_per_subap * num_lenslets 
    pupil_diameter = telescope_diameter

    # make a telescope pupil
    pupil_grid = make_pupil_grid(num_pupil_pixels, diameter=pupil_diameter)
    pupil_aperture_generator = make_obstructed_circular_aperture(telescope_diameter, 
                                                                central_obscuration_ratio, 
                                                                num_spiders=num_spiders, 
                                                                spider_width=spider_width)
    telescope_pupil = evaluate_supersampled(pupil_aperture_generator, pupil_grid, 16)

    # make an atmosphere
    fried_parameter = seeing_to_fried_parameter(seeing) # [m]
    Cn_squared = Cn_squared_from_fried_parameter(fried_parameter, 500e-9)
    layer = InfiniteAtmosphericLayer(pupil_grid, Cn_squared, outer_scale, velocity)

    sci_min, sci_max = wavelength_sci - sci_bandwidth/2, wavelength_sci + sci_bandwidth/2
    wfs_min, wfs_max = wavelength_wfs - wfs_bandwidth/2, wavelength_wfs + wfs_bandwidth/2

        # make a grid for the SC focal plane & add two detectors
    spatial_res = 1.22 * sci_max * focal_length / telescope_diameter # all variables use meters
    num_airy = fov_sci/platescale / spatial_res
    q = math.ceil(spatial_res / sci_cam_px_size) # q = number of pixels between rings, num_airy = number of rings that should be visible
    focal_grid_sci = make_focal_grid(q=q, num_airy=num_airy, spatial_resolution=spatial_res)
    n_pixels_sci = len(focal_grid_sci.coords[0])
    science_detector_size = np.sqrt(n_pixels_sci) * sci_cam_px_size

    if target_source[-4:] == '.png':
        wavefronts_target = get_wavefronts_from_image(target_source, focal_grid_sci, pupil_grid, telescope_pupil)
    else:
        wavefronts_target = get_wavefronts_from_fits(target_source, area_aperture, telescope_pupil, simulation_id)

    prop = FraunhoferPropagator(pupil_grid, focal_grid_sci, focal_length=5.6)
    # imshow_field(prop(wavefronts_target[0]).intensity)
    # plt.show()


    target_wlens = np.array([wf.wavelength for wf in wavefronts_target])
    target_powers = np.array([wf.total_power for wf in wavefronts_target]) 

    wfs_min, wfs_max = wavelength_wfs - wfs_bandwidth/2, wavelength_wfs + wfs_bandwidth/2
    wfs_wavefronts = np.array(wavefronts_target)[(target_wlens > wfs_min) & (target_wlens < wfs_max)]
    for wf in wfs_wavefronts:
        wf.total_power = wfs_beamsplit*np.array(target_powers)[(target_wlens == wf.wavelength)][0]
    wfs_wls = [wf.wavelength*1e10 for wf in wfs_wavefronts] # convert wavelengths back to angstrom for plotting
    wfs_power = [wf.total_power for wf in wfs_wavefronts]

    target_copy2 = wavefronts_target.copy()
    sci_wavefronts = np.array(wavefronts_target)[(target_wlens > sci_min) & (target_wlens < sci_max)]
    for wf in sci_wavefronts:
        if (wf.wavelength < wfs_max) and (wf.wavelength > wfs_min):
            wf.total_power = (1 - wfs_beamsplit)*np.array(target_powers)[(target_wlens == wf.wavelength)][0]
    sci_wls = np.array([wf.wavelength*1e10 for wf in sci_wavefronts]) # convert wavelengths back to angstrom for plotting
    sci_power = np.array([wf.total_power for wf in sci_wavefronts])

    h = 6.626e-27 # erg * s
    c = 3e5 * 1e13 # Angstrom / s

    # set some more WFS parameters
    wlen_to_scale = wfs_min + wfs_bandwidth*wavelength_percentile_to_scale
    diff_limit_telescope = 1.22 * wlen_to_scale / telescope_diameter * 206265 # [arcsec]
    spot_size_telescope = diff_limit_telescope / platescale # [m]
    diff_limit_wfs = diff_limit_telescope * num_lenslets # [arcsec]
    n_px_one_spot = round(diff_limit_wfs / px_scale,0)
    spot_size_wfs = n_px_one_spot * px_size # [m]
    wfs_diameter = px_size * px_per_subap * num_lenslets 
    lenslet_diameter = px_size * px_per_subap 
    magnification_pupil = wfs_diameter / telescope_diameter 
    f_coll = focal_length * magnification_pupil 
    magnification_focal = spot_size_wfs / spot_size_telescope
    f_wfs = f_coll * magnification_focal
    f_number = f_wfs / wfs_diameter
    platescale_wfs = platescale / magnification_focal
    magnifier = Magnifier(magnification_pupil)

    diff_limit_sc = 1.22 * (wavelength_sci + sci_bandwidth/2) / telescope_diameter * 206265
    wfs_exp_time = 1 / operation_freq
    pwfs_exp_time = 1 / operation_freq_pwfs

    focal_grid_wfs = pupil_grid.scaled(magnification_pupil)
    n_pixels_wfs = int(len(focal_grid_wfs.coords[0]))
    
    num_pupil_pixels_pwfs = (telescope_diameter/fried_parameter) * 2 * px_scale_pwfs * 2
    pwfs_grid = make_pupil_grid(num_pupil_pixels, 2 * pupil_diameter)

    return locals()


def get_wavefronts_from_image(image_source, focal_grid, pupil_grid, telescope_pupil, magnitude=5):
    img = Image.open(image_source).convert("L")
    image = np.asarray(img, dtype=float)

    # Normalize to total brightness = 1
    image /= image.sum()
    # image *= TODO: scale by magnitude
    num_pupil_pixels = int(np.sqrt(len(pupil_grid)))
    #print('crop size: ', focal_grid.shape[0]) # 94
    crop_size = focal_grid.shape[0] #num_pupil_pixels

    ny, nx = image.shape
    #print('image shape: ', ny, nx) # 626x646 or so
    x0 = (nx - crop_size) // 2
    y0 = (ny - crop_size) // 2

    image = image[
        y0:y0 + crop_size,
        x0:x0 + crop_size
    ]
    #print('cropped image shape: ', image.shape) #94x94
    # plt.imshow(image)
    # plt.show()
    image = np.zeros(focal_grid.shape)
    #image[75][75] = 1
    image[75][30] = 1
    # plt.imshow(image)
    # plt.show()

    wavefronts = []

    theta_x = focal_grid.x.reshape(focal_grid.shape)
    theta_y = focal_grid.y.reshape(focal_grid.shape)

    for iy in range(image.shape[0]):
        for ix in range(image.shape[1]):

            brightness = image[iy, ix]

            if brightness <= 0:
                continue
            tx = theta_x[iy, ix]
            ty = theta_y[iy, ix]

            wf = Wavefront(telescope_pupil, wavelength_wfs)

            phase = 2 * np.pi / wavelength_wfs * (
                tx * pupil_grid.x +
                ty * pupil_grid.y
            )

            wf.electric_field *= np.exp(1j * phase)
            wf.electric_field *= np.sqrt(brightness)

            wavefronts.append(wf)
    print(len(wavefronts))
    
    return wavefronts

def get_wavefronts_from_fits(target_source, area_aperture, telescope_pupil, simulation_id):
    config_file = f'configs/config_{simulation_id}.yaml'
    with open(config_file) as f:
            globals().update(yaml.safe_load(f))

        # get transmission/reflectivity files
    target_spectrum_file = Path(target_source) # can be .fits or .csv
    telescope_coating_file = Path(coating_file) # wavelength in [µm]
    ao_coating_file = Path(coating_file)

# get light from target
    hdu = fits.open(target_spectrum_file)
    target_wavelength, target_flux = hdu[1].data['WAVELENGTH'], hdu[1].data['FLUX'] # [angstroms], [FLAM = erg * s^-1 * cm^-2 * angstr^-1]
    target_wavelength, target_flux = target_wavelength[::stepsize_wavefronts_to_sample], target_flux[::stepsize_wavefronts_to_sample]
    target_power = target_flux*area_aperture*1e4 # [erg * s^-1 * angstr^-1]

    # pass it through the system (telescope + AO)
    wavelength_crop1, transmitted_power_telescope = transmitted_power(wavelength = target_wavelength, 
                                                                    in_power = target_power,
                                                                    transmission_file = telescope_coating_file,
                                                                    n_surfaces = n_telescope_mirrors)

    wavelength_crop2, transmitted_power_ao = transmitted_power(wavelength = wavelength_crop1, 
                                                            in_power = transmitted_power_telescope,
                                                            transmission_file = ao_coating_file,
                                                            n_surfaces = n_ao_mirrors)

    # make WFS & SC wavefronts for this target
    wavefronts_target = []
    for i, wlen in enumerate(wavelength_crop2): 
        wf = Wavefront(telescope_pupil, wlen*1e-10) # HCIPy wavefronts take wavelengths in meters only!
        wf.total_power = transmitted_power_ao[i] # [erg * s^-1 * angstr^-1] 
        wavefronts_target.append(wf)

    return wavefronts_target