import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hcipy import imshow_field
import yaml
from secondary_variables import get_secondary_variables

def plot_svd(response_matrix, simulation_id, rcond, suffix):
    svd = np.linalg.svd(response_matrix.transformation_matrix)[1]
    # print(svd) -- if this is empty, maybe the SC is getting no light because it shares bands with the WFS
    norm_svd = svd/svd[0]
    plt.plot(np.arange(0,len(svd)), norm_svd)
    cutoff_index = np.where(norm_svd >= rcond)[0][-1]
    plt.axvline(cutoff_index, linestyle='dashed', color='red')
    plt.yscale("log")
    plt.xlabel('actuator number')
    plt.ylabel('Eigenvalue')
    plt.title('Response matrix SVD with truncation threshold')
    plt.savefig(f'simulations/sim_{simulation_id}/svd_response_matrix_{suffix}.png')
    plt.clf()

def plot_response_matrix(response_matrix, simulation_id, suffix):
    plt.imshow(response_matrix)
    plt.xlabel('slope x, slope y')
    plt.ylabel('actuator')
    if suffix == 'pwfs':   
        plt.gca().set_aspect(100)
    plt.savefig(f'simulations/sim_{simulation_id}/response_matrix_{suffix}.png')
    plt.clf()

def plot_reconstruction_matrix(reconstruction_matrix, simulation_id, suffix):
    plt.imshow(reconstruction_matrix)
    plt.xlabel('slope x, slope y')
    plt.ylabel('actuator')
    if suffix == 'pwfs':
        plt.gca().set_aspect(100)
    plt.savefig(f'simulations/sim_{simulation_id}/reconstruction_matrix_{suffix}.png')
    plt.clf()

def plot_example_reconstruction(shwfs, shwfse, reconstruction_matrix, simulation_id):
    x = shwfs.mla_grid.subset(shwfse.estimation_subapertures).x
    y = shwfs.mla_grid.subset(shwfse.estimation_subapertures).y
    act = 25
    vec = reconstruction_matrix[act]
    centroids = vec.reshape(2, -1)
    dx = centroids[0]
    dy = centroids[1]
    demag = 0.5
    plt.figure(figsize=(6,6))
    plt.title(f"Reconstruction matrix actuator {act}")
    plt.scatter(x, y, s=20, color='k')
    plt.quiver(
        x, y,
        dx*demag, dy*demag,
        angles='xy',
        scale_units='xy',
        scale=1,
        color='red'
    )
    plt.gca().set_aspect('equal')
    plt.xlabel("Pupil x")
    plt.ylabel("Pupil y")
    plt.savefig(f'simulations/sim_{simulation_id}/example_reconstruction.png')
    plt.clf()


def plot_fwhm_evol(times, fwhms, fwhms_pwfs, simulation_id):
    df_fwhm = pd.DataFrame({'time': times, 'fwhm_corr': fwhms, 'fwhm_pwfs': fwhms_pwfs}) #, 'fwhm_uncorr': fwhms_uncorr
    df_fwhm.to_csv(f'simulations/sim_{simulation_id}/fwhm_evol.csv')
    plt.plot(times, fwhms, label='SH corrected')
    plt.plot(times, fwhms_pwfs, label='PY corrected')    
    plt.legend()
    plt.xlabel('time [s]')
    plt.ylabel('FWHM (arcsec)')
    plt.title('FWHM over time as loop closes')
    plt.savefig(f'simulations/sim_{simulation_id}/fwhms_evol.png')
    plt.clf()

def plot_strehl_evol(times, strehls, strehls_pwfs, strehls_uncorr, simulation_id):
    df_evol = pd.DataFrame({'time': times, 'strehl_corr': strehls, 'strehls_py': strehls_pwfs, 'strehl_uncorr': strehls_uncorr})
    df_evol.to_csv(f'simulations/sim_{simulation_id}/strehl_evol.csv')
    plt.plot(times, strehls, label='SH corrected')
    plt.plot(times, strehls_pwfs, label='PY corrected')
    plt.plot(times, strehls_uncorr, label='uncorrected')
    plt.legend()
    plt.xlabel('time [s]')
    plt.ylabel('Strehl')
    plt.title('Strehl value over time as loop closes')
    plt.savefig(f'simulations/sim_{simulation_id}/strehl_evol.png')
    plt.clf()

def plot_dm_flat(counts_ref, shwfs, shwfse, slopes_ref, simulation_id):
    plt.title('SH slopes Flat field')
    im2 = imshow_field(counts_ref)
    plt.quiver(shwfs.mla_grid.subset(shwfse.estimation_subapertures).x,
        shwfs.mla_grid.subset(shwfse.estimation_subapertures).y,
        slopes_ref[0,:], slopes_ref[1,:],
        color='white')
    plt.savefig(f'simulations/sim_{simulation_id}/dm_flat_slope.png')
    plt.clf()

def plot_input_light(simulation_id):
    config_file = f'configs/config_{simulation_id}.yaml'
    with open(config_file) as f:
        globals().update(yaml.safe_load(f))

    globals().update(get_secondary_variables(simulation_id))

    plt.plot(target_wavelength, target_power, label = 'target spectrum')
    plt.plot(wavelength_crop2, transmitted_power_ao, label = 'output AO')
    plt.plot(wavelength_crop1, transmitted_power_telescope, label='output telescope')
    plt.plot(wfs_wls, wfs_power, color='red', label = 'WFS input')
    plt.fill_between(wfs_wls, wfs_power, color = 'red', alpha=0.2)
    plt.plot(sci_wls, sci_power, color = 'purple', label = 'Sciencecam input')
    plt.fill_between(sci_wls, sci_power, color = 'purple', alpha=0.2)
    plt.legend()
    plt.xlabel('wavelength [angstr.]')
    plt.ylabel('power [erg/s/angstr./m^2]')
    plt.xlim(3500, 15000)
    plt.ylim(0,1.5e-7)
    plt.savefig(f'simulations/sim_{simulation_id}/input_light.png')
    plt.clf()