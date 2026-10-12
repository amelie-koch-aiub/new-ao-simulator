import yaml 
import math
from pathlib import Path
import astropy.io.fits as fits
import numpy as np
from hcipy import *
from transmitted_power import transmitted_power
from secondary_variables import get_secondary_variables
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.animation import FFMpegWriter
from plotting import *


def make_deformable_mirror(simulation_id, wfs_type='sh'):

    config_file = f'configs/config_{simulation_id}.yaml'
    with open(config_file) as f:
        globals().update(yaml.safe_load(f))

    globals().update(get_secondary_variables(simulation_id))

    if wfs_type == 'sh':
        int_matrix_type = interaction_matrix_type
    else:
        int_matrix_type = interaction_matrix_type_pwfs
    
    if int_matrix_type == 'zonal':
        num_actuators = num_lenslets + 1 #(telescope_diameter/fried_parameter)**2 #len(shwfse.estimation_subapertures) +1 
        num_actuators = int(round(num_actuators*(1 + actuator_oversampling),0))
        actuator_spacing = pupil_diameter / (num_actuators-1)
        actuator_grid = make_actuator_positions(num_actuators,
                                                actuator_spacing=actuator_spacing)
        influence_functions = make_gaussian_influence_functions(pupil_grid,
                                                                num_actuators_across_pupil=num_actuators,
                                                                actuator_spacing=actuator_spacing)

    elif int_matrix_type == 'modal':
        num_actuators = (telescope_diameter/fried_parameter)**2
        num_actuators = int(round(num_actuators*(1 + actuator_oversampling),0))
        num_modes = num_actuators  
        influence_functions = make_zernike_basis(num_modes=num_modes, 
                                    D=telescope_diameter,
                                    grid=pupil_grid, 
                                    use_cache=False) # btw: harmonic disk basis uses fourier bessel basis functions
    else:
        print(f'Error: interaction matrix type not recognized, must be zonal or modal, you gave: {int_matrix_type}')

    norm_modes = [mode / np.ptp(mode) for mode in influence_functions]
    dm_modes = ModeBasis(norm_modes, pupil_grid)
    
    return DeformableMirror(dm_modes), norm_modes, num_actuators

def dm_calibration(deformable_mirror, norm_modes, camera, shwfs, shwfse, simulation_id):

    config_file = f'configs/config_{simulation_id}.yaml'
    with open(config_file) as f:
        globals().update(yaml.safe_load(f))

    globals().update(get_secondary_variables(simulation_id))

    if plotting['calibration_animation']: 
        response_matrix = cal_with_animation(deformable_mirror, norm_modes, shwfs, shwfse, camera, simulation_id)
    else:
        response_matrix = cal_without_animation(deformable_mirror, norm_modes, shwfs, shwfse, camera, simulation_id)
    
    # invert response matrix
    if inversion_method == 'tikhonov':
        reconstruction_matrix = inverse_tikhonov(response_matrix.transformation_matrix, rcond=rcond)
    elif inversion_method == 'truncation':
        reconstruction_matrix = inverse_truncated(response_matrix.transformation_matrix, rcond=rcond)
    else:
        print('Error: Inversion method not recognized.')

    np.save(f"simulations/sim_{simulation_id}/SH_response_matrix.npy", response_matrix)
    np.save(f"simulations/sim_{simulation_id}/SH_reconstruction_matrix.npy", reconstruction_matrix)

    if plotting['svd']:
        plot_svd(response_matrix, simulation_id, rcond, suffix = 'sh')
    if plotting['response_matrix']:
        plot_response_matrix(response_matrix, simulation_id, suffix = 'sh')
    if plotting['reconstruction_matrix']:
        plot_reconstruction_matrix(reconstruction_matrix, simulation_id, suffix = 'sh')
    if plotting['example_reconstruction']:
        plot_example_reconstruction(shwfs, shwfse, reconstruction_matrix, simulation_id)

    return response_matrix, reconstruction_matrix

def dm_calibration_pwfs(deformable_mirror, norm_modes, camera, pwfs, image_ref_pwfs, simulation_id):

    config_file = f'configs/config_{simulation_id}.yaml'
    with open(config_file) as f:
        globals().update(yaml.safe_load(f))

    globals().update(get_secondary_variables(simulation_id))

    if plotting['calibration_animation_pwfs']: 
        response_matrix = cal_with_animation_pwfs(deformable_mirror, norm_modes, pwfs, camera, image_ref_pwfs, simulation_id)
    else:
        response_matrix = cal_without_animation_pwfs(deformable_mirror, norm_modes, pwfs, camera, image_ref_pwfs, simulation_id)
    
    # invert response matrix
    if inversion_method_pwfs == 'tikhonov':
        reconstruction_matrix = inverse_tikhonov(response_matrix.transformation_matrix, rcond=rcond_pwfs)
    elif inversion_method_pwfs == 'truncation':
        reconstruction_matrix = inverse_truncated(response_matrix.transformation_matrix, rcond=rcond_pwfs)
    else:
        print('Error: Inversion method not recognized.')

    np.save(f"simulations/sim_{simulation_id}/PY_response_matrix.npy", response_matrix)
    np.save(f"simulations/sim_{simulation_id}/PY_reconstruction_matrix.npy", reconstruction_matrix)

    if plotting['svd']:
        plot_svd(response_matrix, simulation_id, rcond_pwfs, suffix = 'pwfs')
    if plotting['response_matrix']:
        plot_response_matrix(response_matrix, simulation_id, suffix='pwfs')
    if plotting['reconstruction_matrix']:
        plot_reconstruction_matrix(reconstruction_matrix, simulation_id, suffix='pwfs')
    # if plotting['example_reconstruction']:
    #     plot_example_reconstruction(shwfs, shwfse, reconstruction_matrix, simulation_id)

    return response_matrix, reconstruction_matrix


def cal_with_animation(deformable_mirror, norm_modes, shwfs, shwfse, camera, simulation_id):
    config_file = f'configs/config_{simulation_id}.yaml'
    with open(config_file) as f:
        globals().update(yaml.safe_load(f))

    globals().update(get_secondary_variables(simulation_id))

    response_matrix = []

    wf = wavefronts_target[int(round(len(wavefronts_target)/2,0))]
    wf.total_power = 1
    E_photon = h * c / (wf.wavelength*1e10)

    # Set up animation
    fig = plt.figure(figsize=(10, 6))
    writer = FFMpegWriter(
        fps=2,
        metadata={"title": "DM Interaction Matrix Calibration"},
        bitrate=1800
    )
    interaction_matrix_output_file = f'simulations/sim_{simulation_id}/dm_calibration.mp4'
    deformable_mirror.flatten()
    with writer.saving(fig, interaction_matrix_output_file, dpi=100):

        for i in tqdm(range(len(norm_modes))):
            slope = 0
        
            # Probe the phase response
            amps = [-probe_amp, probe_amp]
            for amp in amps:
                deformable_mirror.flatten()
                deformable_mirror.actuators[i] = amp
        
                dm_wf = deformable_mirror.forward(wf)
                wfs_wf = shwfs(magnifier(dm_wf))

                camera.integrate(wfs_wf, 1e-3)
                image = camera.read_out()            
                slopes = shwfse.estimate([image])
                slope += amp * slopes / np.var(amps)
        
            response_matrix.append(slope.ravel())
        
            plt.clf()
            #plt.suptitle('Mode %d / %d' % (i + 1, num_actuators), y=0.87)
        
            plt.subplot(1,2,1)
            plt.title('DM surface')
            im1 = imshow_field(deformable_mirror.surface, cmap='RdBu', mask=telescope_pupil)
        
            plt.subplot(1,2,2)
            plt.title('SH spots (slopes not to scale)')
            im2 = imshow_field(image)
            plt.quiver(shwfs.mla_grid.subset(shwfse.estimation_subapertures).x,
                shwfs.mla_grid.subset(shwfse.estimation_subapertures).y,
                slope[0,:], slope[1,:],
                color='white')
        
            writer.grab_frame()

    response_matrix = ModeBasis(response_matrix)
    plt.close(fig)
    print(f"Animation saved to: {interaction_matrix_output_file}")

    return response_matrix

def cal_without_animation(deformable_mirror, norm_modes, shwfs, shwfse, camera, simulation_id):
    config_file = f'configs/config_{simulation_id}.yaml'
    with open(config_file) as f:
        globals().update(yaml.safe_load(f))

    globals().update(get_secondary_variables(simulation_id))

    wf = wavefronts_target[int(round(len(wavefronts_target)/2,0))]
    wf.total_power = 1
    E_photon = h * c / (wf.wavelength*1e10)
    response_matrix = []

    for i in tqdm(range(len(norm_modes))):
        slope = 0
    
        # Probe the phase response
        amps = [-probe_amp, probe_amp]
        for amp in amps:
            deformable_mirror.flatten()
            deformable_mirror.actuators[i] = amp
    
            dm_wf = deformable_mirror.forward(wf)
            wfs_wf = shwfs(magnifier(dm_wf))

            camera.integrate(wfs_wf, 1e-3)
            image = camera.read_out()            
            slopes = shwfse.estimate([image])
            slope += amp * slopes / np.var(amps)
    
        response_matrix.append(slope.ravel())

    response_matrix = ModeBasis(response_matrix)

    return response_matrix

def cal_with_animation_pwfs(deformable_mirror, norm_modes, pwfs, camera, image_ref_pwfs, simulation_id):
    config_file = f'configs/config_{simulation_id}.yaml'
    with open(config_file) as f:
        globals().update(yaml.safe_load(f))

    globals().update(get_secondary_variables(simulation_id))

    response_matrix = []

    wf = wavefronts_target[int(round(len(wavefronts_target)/2,0))]
    wf.total_power = 1
    E_photon = h * c / (wf.wavelength*1e10)

    # Set up animation
    fig = plt.figure(figsize=(10, 6))
    writer = FFMpegWriter(
        fps=2,
        metadata={"title": "DM Interaction Matrix Calibration"},
        bitrate=1800
    )
    interaction_matrix_output_file = f'simulations/sim_{simulation_id}/PY_dm_calibration.mp4'
    deformable_mirror.flatten()
    with writer.saving(fig, interaction_matrix_output_file, dpi=100):

        for i in tqdm(range(len(norm_modes))):
            slope = 0
        
            # Probe the phase response
            amps = [-probe_amp_pwfs, probe_amp_pwfs]
            for amp in amps:
                deformable_mirror.flatten()
                deformable_mirror.actuators[i] = amp
                wfs_wf = pwfs.forward(deformable_mirror(wf))
                camera.integrate(wfs_wf, 1e-3)
                image = camera.read_out()            
                image /= np.sum(image)
                slope += (image-image_ref_pwfs)/(2 * amp)
                # TODO fix this according to tutorial
            response_matrix.append(slope)

            plt.clf()
            #plt.suptitle('Mode %d / %d' % (i + 1, num_actuators), y=0.87)
        
            plt.subplot(1,2,1)
            plt.title('DM surface')
            im1 = imshow_field(deformable_mirror.surface, cmap='RdBu', mask=telescope_pupil)
        
            plt.subplot(1,2,2)
            plt.title('WFS response')
            im2 = imshow_field(image)
        
            writer.grab_frame()

    response_matrix = ModeBasis(response_matrix)
    plt.close(fig)
    print(f"Animation saved to: {interaction_matrix_output_file}")

    return response_matrix

def cal_without_animation_pwfs(deformable_mirror, norm_modes, pwfs, camera, image_ref_pwfs, simulation_id):
    config_file = f'configs/config_{simulation_id}.yaml'
    with open(config_file) as f:
        globals().update(yaml.safe_load(f))

    globals().update(get_secondary_variables(simulation_id))

    response_matrix = []
    wf = wavefronts_target[int(round(len(wavefronts_target)/2,0))]
    wf.total_power = 1
    E_photon = h * c / (wf.wavelength*1e10)

    for i in tqdm(range(len(norm_modes))):
        slope = 0
        # Probe the phase response
        amps = [-probe_amp_pwfs, probe_amp_pwfs]
        for amp in amps:
            deformable_mirror.flatten()
            deformable_mirror.actuators[i] = amp

            wfs_wf = pwfs.forward(deformable_mirror(wf))

            camera.integrate(wfs_wf, 1e-3)
            image = camera.read_out()            
            image /= np.sum(image)

            slope += (image-image_ref_pwfs)/(2 * amp)
        # print('adding pwfs slope of length ', len(slope))
        # print('image shape pwfs ',image.shape)
        response_matrix.append(slope)

    response_matrix = ModeBasis(response_matrix)

    return response_matrix
