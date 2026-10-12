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

def ao_loop_with_animation(deformable_mirror, 
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
                           simulation_id):
    
    config_file = f'configs/config_{simulation_id}.yaml'
    with open(config_file) as f:
        globals().update(yaml.safe_load(f))
    globals().update(get_secondary_variables(simulation_id))

    grid = shwfs.mla_grid 

    layer.reset()
    deformable_mirror.flatten()
    deformable_mirror_pwfs.flatten()

    fig = plt.figure(figsize=(12, 10))
    writer = FFMpegWriter(
        fps=int(playback_speed / t_atmos),
        metadata={"title": "Closing the Loop with turbulent atmosphere"},
        bitrate=1800
    )
    output_file = f"simulations/sim_{simulation_id}/ao_animation.mp4"

    strehls = []
    strehls_uncorr = []
    strehls_pwfs = []

    fwhms = []
    fwhms_pwfs = []
    fwhms_uncorr = []

    mfgs_all = []
    mfgs_all_pwfs = []
    mfgs_uncorr_all = []


    wfs_energy = 0
    wfs_photon_image = 0
    counts_per_angstr_turb = np.zeros((len(wfs_wavefronts), n_pixels_wfs))
    photon_counts_turb = Field(np.zeros(n_pixels_wfs), focal_grid_wfs)

    counts_per_angstr_turb_pwfs = np.zeros((len(wfs_wavefronts), n_pixels_wfs))
    photon_counts_turb_pwfs = Field(np.zeros(n_pixels_wfs), pwfs_grid)
    #print(len(counts_per_angstr_turb[0]), n_pixels_wfs, len(photon_counts_turb))

    actuator_command = deformable_mirror.actuators
    actuator_command_pwfs = deformable_mirror_pwfs.actuators
    
    fails = 0
    with writer.saving(fig, output_file, dpi=100):
        
        for timestep in tqdm(range(num_iterations)):

            # atmosphere evolution
            layer.t = timestep * t_atmos
            phase_screen_phase = layer.phase_for(wavelength_wfs) # [rad]
            phase_screen_opd = phase_screen_phase * (wavelength_wfs / (2 * np.pi)) * 1e6 # [µm]
        
            # WFS image capture
            for i, wf in enumerate(wfs_wavefronts):

                E_photon = h * c / (wf.wavelength*1e10) # [erg]
                camera.integrate(shwfs(magnifier(deformable_mirror(layer(wf)))), t_atmos) 
                camera_pwfs.integrate(pwfs.forward(deformable_mirror_pwfs(layer(wf))), t_atmos)
                wfs_data = camera.read_out()
                pwfs_data = camera_pwfs.read_out()
                counts_per_angstr_turb[i] = wfs_data / E_photon
                counts_per_angstr_turb_pwfs[i] = pwfs_data / np.sum(pwfs_data)
                

            for i in range(n_pixels_wfs): # for each pixel, integrate photon counts/intensity across all wavelengths
                #print(f'{i=}, {photon_counts_turb.shape=}')
                photon_counts_turb[i] += integrate.simpson(counts_per_angstr_turb[:, i], wfs_wls)
                photon_counts_turb_pwfs[i] += integrate.simpson(counts_per_angstr_turb_pwfs[:, i], wfs_wls)
                

            if int(layer.t*10000) % int(wfs_exp_time*10000) == 0:
                wfs_photon_image_to_plot = photon_counts_turb
                photon_counts_turb = Field(np.zeros(n_pixels_wfs), focal_grid_wfs)

            if int(layer.t*10000) % int(pwfs_exp_time*10000) == 0:
                pwfs_photon_image_to_plot = photon_counts_turb_pwfs
                photon_counts_turb_pwfs = Field(np.zeros(n_pixels_wfs), pwfs_grid)

            # Science image capture (corrected and uncorrected)
            psf_ref = 0
            for wf in sci_wavefronts:
                sci_camera.integrate(prop(layer(wf)), t_atmos) # uncorrected
                sci_camera2.integrate(prop(deformable_mirror(layer(wf))), t_atmos) # corrected
                sci_camera_pwfs.integrate(prop(deformable_mirror_pwfs(layer(wf))), t_atmos)
                psf_ref += prop.forward(wf).power*sci_exp_time #*prop.forward(wf).wavelength*1e10*5.03e7

            if int(layer.t*1000) % int(sci_exp_time*1000) == 0: # modulo behaves weird with decimals
                
                psf_uncorr = sci_camera.read_out()
                psf_corr = sci_camera2.read_out()
                psf_corr_pwfs = sci_camera_pwfs.read_out()
            
                strehl = float(get_strehl_from_focal(psf_corr, psf_ref))
                strehl_uncorr = float(get_strehl_from_focal(psf_uncorr, psf_ref))
                strehl_pwfs = float(get_strehl_from_focal(psf_corr_pwfs, psf_ref))

                if timestep in plotting['psf_profiles']:
                    plot = True
                else:
                    plot = False

                x_fwhm = find_gaussian_fwhm(psf_corr, psf_uncorr, platescale, simulation_id, timestep, axis='x', plot=plot, fails=fails)
                y_fwhm = find_gaussian_fwhm(psf_corr, psf_uncorr, platescale, simulation_id, timestep, axis='y', plot=plot, fails=fails)

                x_fwhm_pwfs = find_gaussian_fwhm(psf_corr_pwfs, psf_uncorr, platescale, simulation_id, timestep, axis='x', plot=plot, fails=fails)
                y_fwhm_pwfs = find_gaussian_fwhm(psf_corr_pwfs, psf_uncorr, platescale, simulation_id, timestep, axis='y', plot=plot, fails=fails)

                if np.inf in [x_fwhm, x_fwhm, x_fwhm_pwfs, x_fwhm_pwfs]:
                    fails += 1
                
                fwhm = np.mean([x_fwhm, y_fwhm]) #diff_limit_sc/strehl # arcsec
                fwhm_pwfs = np.mean([x_fwhm_pwfs, y_fwhm_pwfs])

                mfgs_corr = mfgs(psf_corr)
                mfgs_corr_pwfs = mfgs(psf_corr_pwfs)

            strehls.append(strehl)
            strehls_uncorr.append(strehl_uncorr)
            strehls_pwfs.append(strehl_pwfs)
            fwhms.append(fwhm)
            fwhms_pwfs.append(fwhm_pwfs)
            mfgs_all.append(mfgs_corr)
            mfgs_all_pwfs.append(mfgs_corr_pwfs)
                        
            # Calculate slopes from WFS image
            if include_photon_noise:
                wfs_photon_image_to_plot = large_poisson(wfs_photon_image_to_plot).astype('float')
                pwfs_photon_image_to_plot = large_poisson(pwfs_photon_image_to_plot).astype('float')

            slopes = shwfse.estimate([wfs_photon_image_to_plot + 1e-20])
            slopes -= slopes_ref
            slopes_copy = slopes
            slopes = slopes.ravel()

            diff_image = pwfs_photon_image_to_plot - image_ref_pwfs
        
            # Set DM actuators (at iteration 0 do nothing. One loop delay) 
            if latency == 1:
                deformable_mirror.actuators = actuator_command
                actuator_command = (1 - leakage) * deformable_mirror.actuators - gain * reconstruction_matrix_sh.dot(slopes)
                deformable_mirror_pwfs.actuators = actuator_command_pwfs
                actuator_command_pwfs = (1-leakage) * deformable_mirror_pwfs.actuators - gain_pwfs * reconstruction_matrix_pwfs.dot(diff_image)
            else:
                deformable_mirror.actuators = (1 - leakage) * deformable_mirror.actuators - gain * reconstruction_matrix_sh.dot(slopes)
                deformable_mirror_pwfs.actuators = (1-leakage) * deformable_mirror_pwfs.actuators - gain_pwfs * reconstruction_matrix_pwfs.dot(diff_image)

            residual_error = 2*deformable_mirror.surface*1e6 + (phase_screen_opd-np.mean(phase_screen_opd))
            residual_error_pwfs = 2*deformable_mirror_pwfs.surface*1e6 + (phase_screen_opd-np.mean(phase_screen_opd))
            

            # plot: atmosphere, DM, WFS image, corr focal image, uncorr focal image
            if timestep % 1 == 0: # can make this plotting frequency more coarse if needed
                plt.clf()
        
                plt.suptitle(f'Gain: {gain}  PSF at $\\lambda$={round(wavelength_sci*1e6,2)}µm  WFS at $\\lambda$={round(wavelength_wfs*1e6,2)}µm  DM stroke: {round(probe_amp*1e6,2)}µm  WFS freq = {operation_freq}Hz  t = {round(layer.t,3)}  S={round(strehl,2)} (uncorr: {round(strehl_uncorr,2)}', y = 0.92)
                plt.subplots_adjust(wspace=0.5) 
                
                plt.subplot(3,3,1)
                plt.title('DM OPD [$\\mu$m]')
                imshow_field(-2*deformable_mirror.surface * 1e6, cmap='RdBu', vmin=-2, vmax=2, mask=telescope_pupil)
                plt.colorbar()
                plt.xlabel('x [m]')
                plt.ylabel('y [m]')

                plt.subplot(3,3,2)
                plt.title('PY-WFS DM OPD [$\\mu$m]')
                imshow_field(-2*deformable_mirror_pwfs.surface * 1e6, cmap='RdBu', vmin=-2, vmax=2, mask=telescope_pupil)
                plt.colorbar()
                plt.xlabel('x [m]')
                plt.ylabel('y [m]')
        
                plt.subplot(3,3,3)
                plt.title('SH image [counts]')
                imshow_field(wfs_photon_image_to_plot, vmin=0, vmax=np.max(wfs_photon_image_to_plot), grid_units=1e-3)
                plt.colorbar()
                for p in grid.points:
                    d = (grid.x[1]-grid.x[0])*1e3
                    rect = plt.Rectangle(p*1e3 - d/ 2, d, d, linewidth=1, edgecolor=colors.red, facecolor='none')
                    plt.gca().add_patch(rect)
                plt.xlabel('x [mm]')
                plt.ylabel('y [mm]')

                plt.subplot(3,3,9)
                plt.title('PY-WFS image [TBD :)]')
                imshow_field(pwfs_photon_image_to_plot, vmin=0, vmax=np.max(pwfs_photon_image_to_plot), grid_units=1e-3)
                plt.colorbar()
                plt.xlabel('x [mm]')
                plt.ylabel('y [mm]')
        
                plt.subplot(3,3,4)
                plt.title(f'SH-Corr. PSF $\\mu$m [log]')
                imshow_field(np.log10(psf_corr/psf_corr.max()), vmin=-5, vmax=0, grid_units=1e-6) 
                plt.colorbar()
                plt.xlabel('x [µm]')
                plt.ylabel('y [µm]')

                plt.subplot(3,3,5)
                plt.title(f'PY-Corr. PSF $\\mu$m [log]')
                imshow_field(np.log10(psf_corr_pwfs/psf_corr_pwfs.max()), vmin=-5, vmax=0, grid_units=1e-6) 
                plt.colorbar()
                plt.xlabel('x [µm]')
                plt.ylabel('y [µm]')

                plt.subplot(3, 3, 6)
                plt.title(f'Residual SH-WF Error [µm]')
                imshow_field(residual_error, vmin=-0.3, vmax=0.3, cmap='RdBu', mask=telescope_pupil) 
                plt.colorbar()
                plt.xlabel('x [m]')
                plt.ylabel('y [m]')

                plt.subplot(3, 3, 7)
                plt.title(f'Residual PY-WF Error [µm]')
                imshow_field(residual_error_pwfs, vmin=-0.3, vmax=0.3, cmap='RdBu', mask=telescope_pupil) 
                plt.colorbar()
                plt.xlabel('x [m]')
                plt.ylabel('y [m]')

                plt.subplot(3,3,8)
                plt.title('Atmosph. OPD [$\\mu$m]')
                imshow_field(phase_screen_opd-np.mean(phase_screen_opd), vmin=-2, vmax=2, cmap='RdBu')
                plt.colorbar()
                plt.xlabel('x [m]')
                plt.ylabel('y [m]')
                
            writer.grab_frame()
            
    plt.close(fig)
    print(f"Animation saved to: {output_file}")

    # calculate average Strehls
    strehl_ao = np.mean(strehls)
    strehl_ao_std = np.std(strehls)
    strehl_noao = np.mean(strehls_uncorr)

    strehl_pwfs = np.mean(strehls_pwfs)
    strehl_pwfs_std = np.std(strehls_pwfs)
    
    avg_PSF_FWHM = np.mean(fwhms)
    std_PSF_FWHM = np.std(fwhms)
    min_PSF_FWHM = np.min(fwhms)
    max_PSF_FWHM = np.max(fwhms)

    avg_PSF_FWHM_pwfs = np.mean(fwhms_pwfs)
    std_PSF_FWHM_pwfs = np.std(fwhms_pwfs)
    min_PSF_FWHM_pwfs = np.min(fwhms_pwfs)
    max_PSF_FWHM_pwfs = np.max(fwhms_pwfs)

    return locals()
    

def ao_loop_without_animation(deformable_mirror, 
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
                                simulation_id):

    # DONT USE THIS (IT IS WRONG)
    print('Warning: This function is not implemented properly for the PY-WFS.')
    
    config_file = f'configs/config_{simulation_id}.yaml'
    with open(config_file) as f:
        globals().update(yaml.safe_load(f))
    globals().update(get_secondary_variables(simulation_id))

    grid = shwfs.mla_grid 

    layer.reset()
    deformable_mirror.flatten()

    strehls = []
    strehls_uncorr = []
    fwhms = []
    fwhms_uncorr = []
    mfgs_all = []

    wfs_energy = 0
    wfs_photon_image = 0
    counts_per_angstr_turb = np.zeros((len(wfs_wavefronts), n_pixels_wfs))
    photon_counts_turb = Field(np.zeros(n_pixels_wfs), focal_grid_wfs)
    actuator_command = deformable_mirror.actuators

        
    for timestep in tqdm(range(num_iterations)):

        # atmosphere evolution
        layer.t = timestep * t_atmos
        phase_screen_phase = layer.phase_for(wavelength_wfs) # [rad]
        phase_screen_opd = phase_screen_phase * (wavelength_wfs / (2 * np.pi)) * 1e6 # [µm]
    
        # WFS image capture
        for i, wf in enumerate(wfs_wavefronts):

            E_photon = h * c / (wf.wavelength*1e10) # [erg]
            camera.integrate(shwfs(magnifier(deformable_mirror(layer(wf)))), t_atmos) 
            wfs_data = camera.read_out()
            counts_per_angstr_turb[i] = wfs_data / E_photon

        for i in range(len(counts_per_angstr_turb[0])): # for each pixel, integrate photon counts across all wavelengths
            photon_counts_turb[i] += integrate.simpson(counts_per_angstr_turb[:, i], wfs_wls)

        if int(layer.t*10000) % int(wfs_exp_time*10000) == 0:
            wfs_photon_image_to_plot = photon_counts_turb
            photon_counts_turb = Field(np.zeros(n_pixels_wfs), focal_grid_wfs)
        

        # Science image capture (corrected and uncorrected)
        psf_ref = 0
        for wf in sci_wavefronts:
            sci_camera.integrate(prop(layer(wf)), t_atmos) # uncorrected
            sci_camera2.integrate(prop(deformable_mirror(layer(wf))), t_atmos) # corrected
            psf_ref += prop.forward(wf).power*sci_exp_time #*prop.forward(wf).wavelength*1e10*5.03e7

        if int(layer.t*1000) % int(sci_exp_time*1000) == 0: # modulo behaves weird with decimals

            psf_uncorr = sci_camera.read_out()
            psf_corr = sci_camera2.read_out()
        
            strehl = float(get_strehl_from_focal(psf_corr, psf_ref))
            strehl_uncorr = float(get_strehl_from_focal(psf_uncorr, psf_ref))

            if timestep in plotting['psf_profiles']:
                plot = True
            else:
                plot = False

            x_fwhm = find_gaussian_fwhm(psf_corr, psf_uncorr, platescale, simulation_id, timestep, axis='x', plot=plot)
            y_fwhm = find_gaussian_fwhm(psf_corr, psf_uncorr, platescale, simulation_id, timestep, axis='y', plot=plot)
            
            fwhm = np.mean([x_fwhm, y_fwhm]) #diff_limit_sc/strehl # arcsec
            mfgs_corr = mfgs(psf_corr)

        strehls.append(strehl)
        strehls_uncorr.append(strehl_uncorr)
        fwhms.append(fwhm)
        mfgs_all.append(mfgs_corr)
                    
        # Calculate slopes from WFS image
        if include_photon_noise:
            wfs_photon_image_to_plot = large_poisson(wfs_photon_image_to_plot).astype('float')
        slopes = shwfse.estimate([wfs_photon_image_to_plot + 1e-20])
        slopes -= slopes_ref
        slopes_copy = slopes
        slopes = slopes.ravel()
    
        # Set DM actuators (at iteration 0 do nothing. One loop delay)
        deformable_mirror.actuators = actuator_command

        actuator_command = (1 - leakage) * deformable_mirror.actuators - gain * reconstruction_matrix.dot(slopes)

    # calculate average Strehls
    strehl_ao = np.mean(strehls)
    strehl_ao_std = np.std(strehls)
    strehl_noao = np.mean(strehls_uncorr)
    
    avg_PSF_FWHM = np.mean(fwhms)
    std_PSF_FWHM = np.std(fwhms)
    min_PSF_FWHM = np.min(fwhms)
    max_PSF_FWHM = np.max(fwhms)
    return locals()
    
