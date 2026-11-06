import os
import nibabel as nib
import numpy as np 
import subprocess
import pandas as pd

'''
Functions used in preprocessing for diffusion data
1. Preprocessing input for multi-shell data and multi-TE data
2. Nordic denoising matlab - both for multishell and multi-TE data
3. Estimate_signal_from_DTI for motion correction on different TEs

Third-party MATLAB code (NOT included in this repository):
- NIFTI_NORDIC.m from NORDIC_Raw (S. Moeller, L. Dowdle; University of Minnesota),
  https://github.com/SteenMoeller/NORDIC_Raw, v1.1, DOI: 10.5281/zenodo.5147408.
  Its licence does not allow redistribution without prior approval, so it must be
  downloaded separately. Cite: Moeller S. et al., NeuroImage 2021;226:117539.
- Tools for NIfTI and ANALYZE image (J. Shen, MATLAB File Exchange, 2014-01-22),
  used by NORDIC to read and write NIfTI files.
Download NIFTI_NORDIC.m and the NIfTI toolbox, then set their locations (mfilepath
and the NIfTI toolbox path) in nordic_denoise() and nordic_denoise_multite() below.
sigma_map.m (own code, included) is found automatically next to this file.
'''
def preproc_input_ms(sub, path2sub):
    for s in sub:
        input_path = os.path.join(path2sub, s, 'ses-1', 'dwi', 'multishell')
        output_path = os.path.join(path2sub, s, 'ses-1', 'preprocessing', 'multishell', 'input')
        os.makedirs(output_path, exist_ok=True)

        # create the b0 AP PA image for top-up input   
        # 1. create the b0 AP image from the dwi image and duplicate the header and affine
        # extract the b0 images from the dwi image
        bval_path = os.path.join(input_path, f'{s}_ses-1_acq-b3_dir-AP_dwi.bval')
        dwi_path = os.path.join(input_path, f'{s}_ses-1_acq-b3_dir-AP_dwi.nii.gz')
        bvec_path = os.path.join(input_path, f'{s}_ses-1_acq-b3_dir-AP_dwi.bvec')

        call = [ 'dwiextract', \
                        '-bzero', \
                        f'-fslgrad {bvec_path} {bval_path}',\
                        dwi_path,\
                        f'{input_path}/{s}_ses-1_acq-b0_dir-AP_dwi.nii',\
                        '-force']
        os.system(' '.join(call)) 

        # extract the first volume from the b0 images *both AP and PA*
        os.system(f"fslroi {os.path.join(input_path, f'{s}_ses-1_acq-b0_dir-AP_dwi.nii')} {os.path.join(input_path, f'{s}_ses-1_acq-b0_dir-AP_dwi.nii')} 0 1")

        # concatenate the b0 AP and PA images to create the topup input
        os.system(f"fslmerge -t {os.path.join(output_path, f'{s}_b0_AP_PA.nii.gz')} {os.path.join(input_path, f'{s}_ses-1_acq-b0_dir-AP_dwi.nii.gz')} {os.path.join(input_path, f'{s}_ses-1_acq-b0_dir-PArevPE_dwi.nii')}")

        # create index file
        # load the dwi_path as a nifti image float64
        dwi_path_multite = os.path.join(input_path, f'{s}_ses-1_acq-b3_dir-AP_dwi.nii.gz')
        dwi = nib.load(dwi_path_multite).get_fdata(dtype=np.float64)
        # create txt file of 1s in a row line, with respect to the number of volumes in the 4th dim then save it as index.txt in the input path
        index = np.ones(dwi.shape[3]).reshape(1, -1)
        np.savetxt(os.path.join(output_path, 'index.txt'), index, fmt='%d', delimiter=' ')

        # copy the bvals and bvecs from input_path to /input
        # os.system(f"cp {bval_path} {output_path + '/input'} && cp {bvec_path} {output_path + '/input'}")


def preproc_input_mt_v2(sub, path2sub):
    for s in sub:
        input_path = os.path.join(path2sub, s, 'ses-1', 'dwi', 'multiTE')
        output_path = os.path.join(path2sub, s, 'ses-1', 'preprocessing', 'multiTE', 'input')
        os.makedirs(output_path, exist_ok=True)
        te_values = ['080', '100', '120', '135']
        # NOTE IF the revPE had more than 2 volumes then extract the first two volumes: 
        os.system(f"fslroi {os.path.join(input_path, f'{s}_ses-1_acq-b0_dir-PArevPE_echo-0080_dwi.nii.gz')} {os.path.join(input_path, f'{s}_ses-1_acq-b0_dir-PArevPE_echo-0080_dwi.nii')} 0 2")

        # create the b0 AP PA image for top-up input   
        for te in te_values:
            bval_path = os.path.join(input_path, f'{s}_ses-1_acq-b5_dir-AP_echo-0{te}_dwi.bval')
            dwi_path = os.path.join(input_path, f'{s}_ses-1_acq-b5_dir-AP_echo-0{te}_dwi.nii.gz')
            bvec_path = os.path.join(input_path, f'{s}_ses-1_acq-b5_dir-AP_echo-0{te}_dwi.bvec')
            
            # Extract the b0 images from the dwi image
            call = ['dwiextract', '-bzero', f'-fslgrad {bvec_path} {bval_path}', dwi_path,
                    f'{input_path}/{s}_ses-1_acq-b0_dir-AP_echo-0{te}_dwi.nii', '-force']
            os.system(' '.join(call)) 

            # Extract the first two volumes from the b0 images *both AP and PA*
            os.system(f"fslroi {os.path.join(input_path, f'{s}_ses-1_acq-b0_dir-AP_echo-0{te}_dwi.nii')} {os.path.join(input_path, f'{s}_ses-1_acq-b0_dir-AP_echo-0{te}_dwi.nii')} 0 2")

            # Concatenate the b0 AP and PA images to create the topup input
            os.system(f"fslmerge -t {os.path.join(output_path, f'{s}_b0_AP_PA_te{te}.nii.gz')} {os.path.join(input_path, f'{s}_ses-1_acq-b0_dir-AP_echo-0{te}_dwi.nii.gz')} {os.path.join(input_path, f'{s}_ses-1_acq-b0_dir-PArevPE_echo-0080_dwi.nii')}")

        # Create index file
        dwi_path_multite = os.path.join(input_path, f'{s}_ses-1_acq-b5_dir-AP_echo-0080_dwi.nii.gz')
        dwi = nib.load(dwi_path_multite).get_fdata(dtype=np.float64)
        index = np.ones(dwi.shape[3]).reshape(1, -1)
        np.savetxt(os.path.join(output_path, 'index.txt'), index, fmt='%d', delimiter=' ')


def nordic_denoise_multite(input_mag, input_phase, input_path, output_path, MATLABEXEC='', s = ''):
    # Calls NIFTI_NORDIC.m (third-party, not included; see the header of this file).
    # EDIT: set mfilepath and the NIfTI toolbox path to your own locations.
    MATLABEXEC = 'matlab'
    mfilepath = '/path/to/NORDIC_denoise_mat/dir'
    ARG = {
        'temporal_phase': 3,
        'NORDIC_patch_overlap': 4,
        'kernel_size_PCA': [4, 4, 4],
        'DIROUT': {output_path},
        'save_residual_matlab': 1
        }
    ARG_str = ', '.join(f"'{k}', {v}" for k, v in ARG.items()) # this is to give the right struct for ARG which is readible by MATLAB
    for mag, phase in zip(input_mag, input_phase):
        full_mag_path = os.path.join(input_path, mag)
        full_phase_path = os.path.join(input_path, phase)
        mag_name = mag.split('.')[0]  # Extract filename without extension
        te_val = mag_name.split('_')[4]
        fn_out = f"{s}_nordicdn_te" + te_val[6:10]
        if not os.path.exists(os.path.join(output_path, fn_out+'.nii')):
            command = f"{MATLABEXEC} -nodesktop -nosplash -r " + \
                    f"\"addpath(genpath('/path/to/MATLAB/NIfTI_20140122/')); " + \
                    f"addpath(genpath('{mfilepath}')); " + \
                    f"NIFTI_NORDIC('{full_mag_path}','{full_phase_path}','{fn_out}', struct({ARG_str})); " + \
                    "quit force;\""
            subprocess.run(command, shell=True)
        else:
            print(f'Denoising has already been run for {s}.')

def nordic_denoise(input_mag, input_phase, input_path, output_path, MATLABEXEC='', s = ''):
    # Calls NIFTI_NORDIC.m (third-party, not included; see the header of this file)
    # and sigma_map.m (included, in the same folder as this file).
    # EDIT: set mfilepath and the NIfTI toolbox path to your own locations.
    MATLABEXEC = 'matlab'
    mfilepath = '/path/to/NORDIC_denoise_mat/dir'
    script_dir = os.path.dirname(os.path.abspath(__file__))  # folder containing sigma_map.m
    input_mag = os.path.join(input_path, input_mag)
    input_phase = os.path.join(input_path, input_phase)
    resid = os.path.join(output_path, f'RESIDUAL{s}_nordicdn.mat')
    denoised = os.path.join(output_path, f'{s}_nordicdn.nii')
    sigma = os.path.join(output_path, f'{s}_SIGMAnordicdn.nii')
    ARG = {
        'temporal_phase': 3,
        'NORDIC_patch_overlap': 5,
        'kernel_size_PCA': [5, 5, 5],
        'DIROUT': {output_path},
        'save_residual_matlab': 1
        }
    ARG_str = ', '.join(f"'{k}', {v}" for k, v in ARG.items()) # this is to give the right struct for ARG which is readible by MATLAB
    fn_out = f"{s}_nordicdn"
    if not os.path.exists(os.path.join(output_path, fn_out+'.nii')):
        command = f"{MATLABEXEC} -nodesktop -nosplash -r " + \
                f"\"addpath(genpath('/path/to/MATLAB/NIfTI_20140122/')); " + \
                f"addpath(genpath('{mfilepath}')); " + \
                f"addpath('{script_dir}'); " + \
                f"NIFTI_NORDIC('{input_mag}','{input_phase}','{fn_out}', struct({ARG_str})); " + \
                f"sigma_map('{input_mag}','{denoised}','{sigma}'); " + \
                "quit force;\""
        subprocess.run(command, shell=True)

    else:
        print(f'Denoising has already been run for {s}.')


def estimate_signal_from_DTI(folder, bval, bvec):
    V1   = nib.load(folder + '/recon_V1.nii.gz').get_fdata()
    V2   = nib.load(folder + '/recon_V2.nii.gz').get_fdata()
    V3   = nib.load(folder + '/recon_V3.nii.gz').get_fdata()
    L1   = nib.load(folder + '/recon_L1.nii.gz').get_fdata()
    L2   = nib.load(folder + '/recon_L2.nii.gz').get_fdata()
    L3   = nib.load(folder + '/recon_L3.nii.gz').get_fdata()
    L3[L3<0] = 0
    S0   = nib.load(folder + '/recon_S0.nii.gz').get_fdata()
    Mask = nib.load(folder + '/mask.nii.gz').get_fdata()

    b = np.loadtxt(bval)
    r = np.loadtxt(bvec)

    [x, y, z, n] = V1.shape
    S = np.zeros((x,y,z, b.shape[0]))

    for i in range(x):
        for j in range(y):
            for k in range(z):
                if Mask[i,j,k] > 0:
                    V1_voxel = V1[i,j,k,:]
                    V2_voxel = V2[i,j,k,:]
                    V3_voxel = V3[i,j,k,:]
                    R = np.dstack((V1_voxel, V2_voxel, V3_voxel)).squeeze()
                    L1_voxel = L1[i,j,k]
                    L2_voxel = L2[i,j,k]
                    L3_voxel = L3[i,j,k]
                    Lvoxel   = np.diag([L1_voxel, L2_voxel, L3_voxel])
                    # Create Diffusion tensor
                    D        = np.matmul(R, np.matmul(Lvoxel, R.T))
                    # Evaluate for all gradient directions
                    ADC      = np.diag(np.dot(r.T, np.dot(D,r)))
                    # Compute the diffusion signal
                    S[i,j,k,:] = S0[i,j,k] * np.exp(-b*ADC)
                #end if
            #end k
        #end j
    #end i
    return S
#end
