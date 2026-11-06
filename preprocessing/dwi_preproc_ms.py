import os
import numpy as np 
import subprocess
from preproc_fn import nordic_denoise
from preproc_fn import preproc_input_ms
import time
''' 
Preprocessing pipeline - multi shell diffusion imaging
Author: Saina Asadi, 2024

Requirements:
    - MATLAB
    - FSL
    - MRtrix3
    - FreeSurfer

Steps:
- Nordic denoising (MATLAB) using both magnitude and phase data. 
- Gibbs ringing correction.
- Top-up correction
- Eddy current and motion correction
- DTI fit 

'''

freesurfer_home = '/usr/local/freesurfer'
# Set up the FreeSurfer environment by explicitly defining the necessary environment variables.
env = {
    'FREESURFER_HOME': freesurfer_home,
    'PATH': f'{freesurfer_home}/bin:{os.environ["PATH"]}',
}

##### 0. Set up the paths #####
mri_synthstrip_path = f'{freesurfer_home}/bin/mri_synthstrip'
path2sub = '*/path/to/sub/dir/*'
subs = ['sub-ctrl**', 'sub-ctrl**']  # list of subject IDs
preproc_input_ms(subs, path2sub) # creates the b0_AP_PA file and index file for topup ADD the acqp file and config manually!

for s in subs:
    main_path = os.path.join(path2sub,f'{s}','ses-1') 
    input_path = os.path.join(main_path,'dwi','multishell')
    output_path = os.path.join(main_path,'preprocessing','multishell/')
    print(f'Processing {s}:')
    os.makedirs(output_path, exist_ok=True) 
    ### 1. NORDIC denoising  #####
    input_mag = f'{s}_ses-1_acq-b3_dir-AP_dwi.nii.gz'
    input_phase = f'{s}_ses-1_acq-b3_dir-AP_part-phase_dwi.nii.gz'
    print('1. Nordic denoising ...')
    start_time = time.time()
    nordic_denoise(input_mag, input_phase, input_path, output_path, MATLABEXEC='matlab', s = f'{s}')
    end_time = time.time()
    exec_time = end_time - start_time
    print(f'Denoising took {exec_time} seconds')

    ##### 2. Gibbs ringing correction  #####
    print('2. Gibbs ringing correction ... ')
    dwi_dn = os.path.join(output_path,f'{s}_nordicdn.nii')
    out_degibbs = f'{output_path}/{s}_nordicdn_dg.nii.gz'
    if not os.path.exists(os.path.join(out_degibbs)):
            call  = [ f'mrdegibbs', \
                f'{dwi_dn}', \
                f'{out_degibbs}', \
            ]
            os.system(' '.join(call))
    else: 
        print(f'Degibbs has already been run.')

    print('Synth-stripping ... brain and brainmask extraction')
    # Synth-stripping after degibbs - extract brain and brain mask
    input_image = os.path.join(output_path, f'{s}_nordicdn_dg.nii.gz')
    output_image = os.path.join(output_path, f'{s}_nordicdn_dg_B0.nii.gz')

    if not os.path.exists(os.path.join(output_image)):
        call  = [ f'fslroi', \
                f'{input_image}', \
                f'{output_image}', \
                f'0 -1 0 -1 0 -1 0 1']
        os.system(' '.join(call))

    mri_synthstrip_cmd = [
        'bash', '-c',
        f'source {freesurfer_home}/SetUpFreeSurfer.sh && {mri_synthstrip_path} '
        f'--image {os.path.join(output_path, f"{s}_nordicdn_dg_B0.nii.gz")} '
        f'--out {os.path.join(output_path, f"{s}_nordicdn_dg_B0_brain.nii.gz")} '
        f'--mask {os.path.join(output_path, f"{s}_nordicdn_dg_B0_brainmask.nii.gz")}',
    ]
    process = subprocess.Popen(mri_synthstrip_cmd, env=env)
    process.wait() 

    ##### 3. Top-up correction #####
    b0 = os.path.join(output_path, 'input',f'{s}_b0_AP_PA.nii')
    acqp = os.path.join(output_path, 'input','acqp.txt')
    config = os.path.join(output_path, 'input','config.cnf')
    topup = os.path.join(output_path, f'{s}_b0_AP_PA_tu.nii.gz')
    topup_fieldcoef = os.path.join(output_path, f'{s}_topup_results')
    print('3. Topup correction...')
    start_time = time.time()
    if not (os.path.exists(topup) or os.path.exists(topup_fieldcoef)):
        call  = [f'topup', \
                f'--imain={b0}', \
                f'--datain={acqp}', \
                f'--config={config}', \
                f'--out={topup_fieldcoef}',\
                f'--iout={topup}']
        os.system(' '.join(call))
    else: 
        print(f'Top-up has already been run.')
    end_time = time.time()
    exec_time_topup = end_time - start_time
    print(f'Topup took {exec_time_topup} seconds')

    print('Extract the brain and brainmask after topup... ')
    # Synth-stripping after top-up - extract brain and brain mask
    input_image = os.path.join(output_path, f'{s}_b0_AP_PA_tu.nii.gz')
    output_image = os.path.join(output_path, f'{s}_b0_AP_PA_tu_B0.nii.gz')
    if not os.path.exists(os.path.join(output_image)):
        call  = [ f'fslroi', \
                f'{input_image}', \
                f'{output_image}', \
                f'0 -1 0 -1 0 -1 0 1']
        os.system(' '.join(call))

    mri_synthstrip_cmd = [
        'bash', '-c',
        f'source {freesurfer_home}/SetUpFreeSurfer.sh && {mri_synthstrip_path} '
        f'--image {os.path.join(output_path, f"{s}_b0_AP_PA_tu_B0.nii.gz")} '
        f'--out {os.path.join(output_path, f"{s}_b0_AP_PA_tu_B0_brain.nii.gz")} '
        f'--mask {os.path.join(output_path, f"{s}_b0_AP_PA_tu_B0_brainmask.nii.gz")}'
    ]
    process = subprocess.Popen(mri_synthstrip_cmd, env=env)
    process.wait()
    #NOTE: you can run this in case the mask needed to be dilated
    input_image = os.path.join(output_path, f'{s}_b0_AP_PA_tu_B0_brainmask.nii.gz')
    output_image = os.path.join(output_path, f'{s}_b0_AP_PA_tu_B0_brainmask_dil.nii.gz')
    if not os.path.exists(os.path.join(output_image)):
        call  = [ f'fslmaths', \
                f'{input_image}', \
                f'-dilM', \
                f'{output_image}']
        os.system(' '.join(call))

    ##### 4. Eddy distortion and motion correction ##### --> 
    dwi  = os.path.join(output_path, f'{s}_nordicdn_dg.nii.gz')
    bvals = os.path.join(input_path, f'{s}_ses-1_acq-b3_dir-AP_dwi.bval')
    bvecs = os.path.join(input_path, f'{s}_ses-1_acq-b3_dir-AP_dwi.bvec')
    mask  = os.path.join(output_path, f'{s}_b0_AP_PA_tu_B0_brainmask.nii.gz')
    acqp  = os.path.join(output_path, 'input', 'acqp.txt')
    index = os.path.join(output_path, 'input', 'index.txt')
    top  = os.path.join(output_path, f'{s}_topup_results')
    eddy = os.path.join(output_path, f'{s}_nordicdn_dg_tu_ec')
    #slicetiming = os.path.join(output_path, 'input', 'slicetiming.txt')
    print('4. Eddy correction...')
    start_time = time.time()
    if not os.path.exists(eddy): 
        call = [f'eddy_cuda10.2', \
            f'--imain={dwi}', \
            f'--mask={mask}', \
            f'--index={index}', \
            f'--acqp={acqp}', \
            f'--bvecs={bvecs}', \
            f'--bvals={bvals}', \
            f'--topup={top}', \
            f'--out={eddy}',
            f'--b_range=200',\
            f'--data_is_shelled']
        os.system(' '.join(call))
    else:
        print(f'Eddy has already been run.')
    end_time = time.time()
    exec_time_eddy = end_time - start_time
    print(f'Eddy took {exec_time_eddy} seconds')
    print('----------------------------------------------------------------------------------------------')
    print(f'Preprocessing for subject {s} is done!')

    # create a {s}_preprocessed folder in the output path to save the preprocessed files and moving the files there 
    if not os.path.exists(os.path.join(output_path, f'{s}_preprocessed')):
        os.makedirs(os.path.join(output_path, f'{s}_preprocessed'))

        os.system(f'cp {os.path.join(output_path, f"{s}_nordicdn_dg_tu_ec.nii.gz")} {os.path.join(output_path, f"{s}_preprocessed")}')
        os.system(f'cp {os.path.join(output_path, f"{s}_b0_AP_PA_tu_B0_brainmask_dil.nii.gz")} {os.path.join(output_path, f"{s}_preprocessed")}')
        os.system(f'cp {os.path.join(input_path, f"{s}_ses-1_acq-b3_dir-AP_dwi.bval")} {os.path.join(output_path, f"{s}_preprocessed")}')
        os.system(f'cp {os.path.join(input_path, f"{s}_ses-1_acq-b3_dir-AP_dwi.bvec")} {os.path.join(output_path, f"{s}_preprocessed")}')
        os.system(f'cp {os.path.join(output_path, f"{s}_SIGMAnordicdn.nii")} {os.path.join(output_path, f"{s}_preprocessed")}')
        os.system(f'cp {os.path.join(output_path, f"{s}_nordicdn_dg_tu_ec.eddy_rotated_bvecs")} {os.path.join(output_path, f"{s}_preprocessed")}')
                                                                        
                                                                        ######## Supplementary steps ######## 
    
       ##### DTI fit (b <= 1000) - for SMI parameter estimation in the white matter #####
    # find all volumes with b <= 1000 (with a small margin for b-values such as 1005); (DTI not valid when including high b-vals)
    print(f'DTI fit for {s}...')
    dti_path = os.path.join(output_path, f'{s}_dti')
    os.makedirs(dti_path, exist_ok=True)
    bval_values = np.loadtxt(bvals)
    bvec_values = np.loadtxt(bvecs)
    low_b_volumes = np.where(bval_values <= 1050)[0]
    low_b_list = ','.join(str(v) for v in low_b_volumes)

    # extract these volumes from the eddy-corrected data
    dwi_b1000 = os.path.join(dti_path, f'{s}_b1000')
    call = [f'fslselectvols', \
            f'-i {eddy}', \
            f'-o {dwi_b1000}', \
            f'--vols={low_b_list}']
    os.system(' '.join(call))

    # save the matching bval and bvec files (fmt = number format: whole numbers for bval, 6 decimals for bvec)
    np.savetxt(f'{dwi_b1000}.bval', [bval_values[low_b_volumes]], fmt='%d')
    np.savetxt(f'{dwi_b1000}.bvec', bvec_values[:, low_b_volumes], fmt='%.6f')

    # DTI fit on the selected volumes
    call = [f'dtifit', \
            f'-k {dwi_b1000}', \
            f'-o {os.path.join(dti_path, f"{s}_dti")}', \
            f'-m {mask}', \
            f'-r {dwi_b1000}.bvec', \
            f'-b {dwi_b1000}.bval']
    os.system(' '.join(call))
    os.system(f'mv {dti_path} {os.path.join(output_path, f"{s}_preprocessed")}')