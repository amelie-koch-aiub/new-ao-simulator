import yaml 
from hcipy import *
from secondary_variables import get_secondary_variables
import numpy as np
import scipy.ndimage as ndimage
import scipy.integrate as integrate
from shackhartmann import MySquareShackHartmannWavefrontSensorOptics

def make_sh(simulation_id):
    config_file = f'configs/config_{simulation_id}.yaml'
    with open(config_file) as f:
        globals().update(yaml.safe_load(f))
    globals().update(get_secondary_variables(simulation_id))

    camera = NoiselessDetector(focal_grid_wfs)

    shwfs = MySquareShackHartmannWavefrontSensorOptics(pupil_grid.scaled(magnification_pupil), 
                                                        f_number, 
                                                        num_lenslets, 
                                                        wfs_diameter)

    # read out reference WFS image
    counts_per_angstr_calm = np.zeros((len(wfs_wavefronts), n_pixels_wfs))
    photon_counts_total = []

    for i, wf in enumerate(wfs_wavefronts):
        E_photon = h * c / (wf.wavelength*1e10) # [erg]
        camera.integrate(shwfs(magnifier(wf)), wfs_exp_time)
        data1 = camera.read_out() # [erg * angstr^-1]
        counts_per_angstr_calm[i] = data1 / E_photon
        photon_counts_total.append(np.sum(data1 / E_photon))
    
    photon_in_cam = integrate.simpson(photon_counts_total, wfs_wls)
    #print(f'Expecting {round(photon_in_cam*1e-6, 2)}e6 photons/s in the WFS at {int(wfs_min*1e10)}-{int(wfs_max*1e10)} Angstroms, {round(photon_in_cam/(num_lenslets)**2,1)} photons per subaperture at {operation_freq}Hz.')

    photon_counts_calm = Field(np.zeros(len(counts_per_angstr_calm[0])), focal_grid_wfs)
    for i in range(len(counts_per_angstr_calm[0])): # loop over pixels, integrate all wavelengths
        photon_counts_calm[i] = integrate.simpson(counts_per_angstr_calm[:, i], wfs_wls)

    # decide which subapertures to use for WFS estimation
    shwfse = ShackHartmannWavefrontSensorEstimator(shwfs.mla_grid, shwfs.micro_lens_array.mla_index)
    fluxes = ndimage.measurements.sum(photon_counts_calm, shwfse.mla_index, shwfse.estimation_subapertures)
    flux_limit = fluxes.max() * cutoff_below
    #print(f'{fluxes.min()=}, {fluxes.max()=}, {flux_limit=}, {fluxes=}')
    estimation_subapertures = shwfs.mla_grid.zeros(dtype='bool')
    estimation_subapertures[shwfse.estimation_subapertures[fluxes > flux_limit]] = True
    shwfse = ShackHartmannWavefrontSensorEstimator(shwfs.mla_grid, shwfs.micro_lens_array.mla_index, estimation_subapertures)

    illuminated_subapertures = len(shwfse.estimation_subapertures)
    print(f'{illuminated_subapertures=}')
    photon_count_per_subap = round(np.mean(ndimage.measurements.sum(photon_counts_calm, shwfse.mla_index, shwfse.estimation_subapertures)),1)

    max_snr = round(np.sqrt(np.mean(ndimage.measurements.maximum(photon_counts_calm, shwfse.mla_index, shwfse.estimation_subapertures))),2)
    avg_snr = round(np.sqrt(np.mean(ndimage.measurements.sum(photon_counts_calm, shwfse.mla_index, shwfse.estimation_subapertures))/(px_per_subap)**2),2)

    pwfs = pwfs = PyramidWavefrontSensorOptics(pupil_grid, pwfs_grid, separation=pupil_diameter, pupil_diameter=telescope_diameter, wavelength_0=wavelength_wfs, q=3)
    camera_pwfs = NoiselessDetector(pwfs_grid)

    return locals()

def get_reference_slopes(shwfs, shwfse, camera, pwfs, camera_pwfs, simulation_id):
    config_file = f'configs/config_{simulation_id}.yaml'
    with open(config_file) as f:
        globals().update(yaml.safe_load(f))
    globals().update(get_secondary_variables(simulation_id))

    wf = wavefronts_target[int(round(len(wavefronts_target)/2,0))]
    E_photon = h * c / (wf.wavelength*1e10) # [erg]
    camera.integrate(shwfs(magnifier(wf)), wfs_exp_time)
    counts_ref = camera.read_out() # [erg * angstr^-1]
    counts_ref /= E_photon
    slopes_ref = shwfse.estimate([counts_ref])

    camera_pwfs.integrate(pwfs(wf), wfs_exp_time)
    image_ref_pwfs = camera_pwfs.read_out()
    image_ref_pwfs /= image_ref_pwfs.sum()

    return slopes_ref, counts_ref, image_ref_pwfs
