from hcipy import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from matplotlib.animation import FFMpegWriter
from tqdm.notebook import tqdm
import astropy.io.fits as fits
from pathlib import Path, PosixPath
import math
from scipy.interpolate import interp1d
import scipy.integrate as integrate
from shackhartmann import MySquareShackHartmannWavefrontSensorOptics
from transmitted_power import transmitted_power
from csv import DictWriter
import pandas as pd
import yaml
import warnings
import os
from matplotlib.patches import Annulus, Circle
from fwhm_util import find_gaussian_fwhm, mfgs, find_nearest
from secondary_variables import get_secondary_variables
from dm_calibration import * #dm_calibration, make_deformable_mirror
from wavefront_sensor import make_sh, get_reference_slopes
from plotting import *
from ao_loop import *
from PIL import Image


warnings.filterwarnings("ignore", category=DeprecationWarning) 

def ao_simulation(simulation_id):

    # load config data
    if not os.path.exists(f'simulations/sim_{simulation_id}'):
        os.mkdir(f'simulations/sim_{simulation_id}')
    config_file = f'configs/config_{simulation_id}.yaml'
    with open(config_file) as f:
        globals().update(yaml.safe_load(f))
    globals().update(get_secondary_variables(simulation_id))

    prop = FraunhoferPropagator(pupil_grid, focal_grid_sci, focal_length=focal_length)

    globals().update(make_sh(simulation_id)) # makes a SH-WFS
    slopes_ref, counts_ref, image_ref_pwfs = get_reference_slopes(shwfs, shwfse, camera, pwfs, camera_pwfs, simulation_id) # can I use photon counts as a unit for slope estimation?

    sci_camera = NoiselessDetector(focal_grid_sci) # uncorrected beam
    sci_camera2 = NoiselessDetector(focal_grid_sci) # AO corrected beam SH
    sci_camera_pwfs = NoiselessDetector(focal_grid_sci) 

    deformable_mirror, norm_modes, num_actuators = make_deformable_mirror(simulation_id, wfs_type = 'sh')
    deformable_mirror_pwfs, norm_modes_pwfs, num_actuators_pwfs = make_deformable_mirror(simulation_id, wfs_type = 'pwfs')


    if (not os.path.exists(f"simulations/sim_{simulation_id}/SH_response_matrix.npy")) or renew_dm_calibration:
        response_matrix_sh, reconstruction_matrix_sh = dm_calibration(deformable_mirror, norm_modes, camera, shwfs, shwfse, simulation_id)
        response_matrix_pwfs, reconstruction_matrix_pwfs = dm_calibration_pwfs(deformable_mirror_pwfs, norm_modes_pwfs, camera, pwfs, image_ref_pwfs, simulation_id)

    else:
        response_matrix_sh = np.load(f"simulations/sim_{simulation_id}/SH_response_matrix.npy")
        reconstruction_matrix_sh = np.load(f"simulations/sim_{simulation_id}/SH_reconstruction_matrix.npy")
        response_matrix_pwfs = np.load(f"simulations/sim_{simulation_id}/PY_response_matrix.npy")
        reconstruction_matrix_pwfs = np.load(f"simulations/sim_{simulation_id}/PY_reconstruction_matrix.npy")

    # print(len(wavefronts_target))
    # deformable_mirror.flatten()
    # for wf in wavefronts_target:
    #     sci_camera.integrate(prop(deformable_mirror(wf)), 1) 
    # data1 = sci_camera.read_out()
    # imshow_field(data1)
    # plt.savefig('data1.png')
    # plt.clf()


    if plotting['ao_animation']:
        globals().update(ao_loop_with_animation(deformable_mirror, 
                           deformable_mirror_pwfs,
                           shwfs, 
                           shwfse, 
                           pwfs,
                           slopes_ref, 
                           image_ref_pwfs,
                           camera, 
                           camera_pwfs,
                           sci_camera, 
                           sci_camera2, 
                           sci_camera_pwfs,
                           prop, 
                           reconstruction_matrix_sh,
                           reconstruction_matrix_pwfs,
                           simulation_id)) 
    else:
        globals().update(ao_loop_without_animation(deformable_mirror, 
                                   shwfs, 
                                   shwfse, 
                                   pwfs,
                                   slopes_ref, 
                                   image_ref_pwfs,
                                   camera, 
                                   camera_pwfs,
                                   sci_camera, 
                                   sci_camera2, 
                                   sci_camera_pwfs,
                                   prop, 
                                   reconstruction_matrix_sh,
                                   reconstruction_matrix_pwfs,
                                   simulation_id)) 
        
    # save results & plots
    new_row = {'simulation_ID': [simulation_id],
            'seeing ["]': [seeing],
            'r0 [m]': [fried_parameter], 
            'v [m/s]': [velocity], 
            'n_actuators': [num_actuators],
            'n_lenslets': [num_lenslets],
            'n_illuminated_subapertures': [len(shwfse.estimation_subapertures)],
            'px_per_subap': [px_per_subap],
            'WFS FOV ["]': [fov_wfs],
            'diff_limit SH ["]': [diff_limit_wfs],
            'WFS platescale ["/m]': [platescale_wfs],
            'SH exp_time [s]': [wfs_exp_time], 
            'WFS_lambda_min [A]': [wfs_min], 
            'WFS_lambda_max [A]': [wfs_max], 
            'SC_exp_time [s]': [sci_exp_time], 
            'SC_lambda_min [A]': [sci_min], 
            'SC_lambda_max [A]': [sci_max], 
            'beamsplit [%]': [wfs_beamsplit], 
            'DM_stroke SH/PY [m]': [probe_amp],
            'gain SH': [gain],
            'gain PY': [gain_pwfs],
            'leakage': [leakage],
            'WFS max_SNR': [max_snr],
            'WFS avg_SNR': [avg_snr],
            'interaction_matrix_type': [interaction_matrix_type],
            'inversion_cutoff': [rcond],
            'inversion_method': [inversion_method],
            'illumination_threshold': [cutoff_below],
            'num_iterations': [num_iterations],
            'stepsize_wavefronts_to_sample': [stepsize_wavefronts_to_sample],
            'actuator_oversampling': [actuator_oversampling],
            'avg_Strehl_noAO': [strehl_noao], 
            'avg_Strehl_SH': [strehl_ao],
            'std_Strehl_SH': [strehl_ao_std],
            'avg_Strehl_PY': [strehl_pwfs],
            'std_Strehl_PY': [strehl_pwfs_std],
            'avg FWHM SH["]': [avg_PSF_FWHM],
            'std FWHM SH ["]': [std_PSF_FWHM],
            'avg FWHM PY ["]': [avg_PSF_FWHM_pwfs],
            'std FWHM PY ["]': [std_PSF_FWHM_pwfs],
            'comments': [comments],
            'loop_closure / quality': ['unassessed']}

    if os.path.exists('ao_performance.csv'):
        df = pd.read_csv('ao_performance.csv')
        tmp_df = pd.DataFrame.from_dict(new_row)
        df = pd.concat([df, tmp_df], ignore_index=True)
    else:
        df = pd.DataFrame.from_dict(new_row)
    df.to_csv('ao_performance.csv', index = False)

    times = np.linspace(0, num_iterations*t_atmos, num_iterations)
    if plotting['fwhm_evol']:
        plot_fwhm_evol(times, fwhms, fwhms_pwfs, simulation_id)
    if plotting['strehl_evol']:
        plot_strehl_evol(times, strehls, strehls_pwfs, strehls_uncorr, simulation_id)
    if plotting['dm_flat']:
        plot_dm_flat(counts_ref, shwfs, shwfse, slopes_ref, simulation_id)
    if plotting['input_light']:
        plot_input_light(simulation_id)

if __name__ == '__main__':
    ids = [88]
    for id in ids:
        ao_simulation(simulation_id=id)
