import os
import nibabel as nib
import numpy as np 
import subprocess
from preproc_fn import nordic_denoise_multite
from preproc_fn import preproc_input_mt_v2
from preproc_fn import estimate_signal_from_DTI
import ants

''' 
Preprocessing pipeline - multi-TE diffusion imaging
Author: Saina Asadi, 2024
	DTI motion correction: Erick J. Canales-Rodriguez

Requirements:
    - MATLAB
    - FSL
    - ANTsPy
    - MRtrix3
    - FreeSurfer

Version: 2.0
    This version will run on subjects acquired with anterior posterior (AP) phase encoding direction only .
Steps:
- Nordic denoising (MATLAB) using both mag and phase data on each TE data. 
- Gibbs ringing correction for each denoised image
- Top-up correction for individual TEs because we have to input each topup result for Eddy
- for Eddy current & motion correction we have 4 outputs from degibbs with 1 field map, we apply eddy on different DWI gibbs corrected image - 
different intensities of TE s don't allow us to run eddy once for merged images. 
- In the end, an additional motion correction step is done to remove the remaining motion for each TE.

'''

freesurfer_home = '/usr/local/freesurfer'
# Set up the FreeSurfer environment by explicitly defining the necessary environment variables.
env = {
    'FREESURFER_HOME': freesurfer_home,
    'PATH': f'{freesurfer_home}/bin:{os.environ["PATH"]}',
}
mri_synthstrip_path = f'{freesurfer_home}/bin/mri_synthstrip'

# set the subjects here: 
path2sub = '*/path/to/sub/dir/*'
subs = ['sub-ctrl**']  # list of subject IDs
te_values = ['080','100','120','135']
preproc_input_mt_v2(subs, path2sub) # creates the b0_AP_PA file and index file for topup, add the acqp.txt and config file in the input folder manually! 

for s in subs:
    main_path = os.path.join(path2sub, f'{s}','ses-1') 
    input_path = os.path.join(main_path,'dwi','multiTE')
    output_path = os.path.join(main_path, 'preprocessing','multiTE/')
    os.makedirs(output_path, exist_ok=True) 
    print(f'Processing subject {s} ...')
    #### 1. Nordic denoising for each TE image #####
    #Input the magnitude and phase data for the multite sequence
    for te in te_values:
        print(f"1. NORDIC denoising, TE = {te}")
        input_mag = [f'{s}_ses-1_acq-b5_dir-AP_echo-0{te}_dwi.nii.gz']
        input_phase = [f'{s}_ses-1_acq-b5_dir-AP_echo-0{te}_part-phase_dwi.nii.gz']
        nordic_denoise_multite(input_mag, input_phase, input_path, output_path, MATLABEXEC='matlab', s = f'{s}')

    ##### 2. Gibbs ringing correction #####
    print('Gibbs ringing correction ... ')
    for te in te_values:
        dwi_dn = os.path.join(output_path, f'{s}_nordicdn_te{te}.nii')
        print(f"Running Degibbs for TE = {te}")
        # Names of the output degibbs image
        out_degibbs = f'{output_path}/{s}_nordicdn_dg_te{te}.nii.gz'
        if not os.path.exists(os.path.join(out_degibbs)):
            call  = [ f'mrdegibbs', \
                f'{dwi_dn}', \
                f'{out_degibbs}', \
            ]
            os.system(' '.join(call))
        else: 
            print(f'2. Degibbs has already been run for echo time: {te}')

    #### 3. Top-up correction #####
    for te in te_values:
        b0 = os.path.join(output_path + '/input', f'{s}_b0_AP_PA_te{te}.nii')
        acqp = os.path.join(output_path + '/input', 'acqp.txt')
        config = os.path.join(output_path + '/input', 'config.cnf') # depending on the n of slices, if it's odd use the config_odd.cnf 
        topup = os.path.join(output_path, f'{s}_b0_AP_PA_tu_te{te}.nii.gz')
        topup_fieldcoef = os.path.join(output_path, f'{s}_topup_results_te{te}')
        print('3. Topup correction for TE = ', te)
        if not (os.path.exists(topup) or os.path.exists(topup_fieldcoef)):
            call = [
                'topup',
                f'--imain={b0}',
                f'--datain={acqp}',
                f'--config={config}',
                f'--out={topup_fieldcoef}',
                f'--iout={topup}'
            ]
            os.system(' '.join(call))
        else: 
            print(f'Top-up has already been run for echo time: {te}')

    print('Extract the brain and brainmask for each TE after top-up ... ')
    # Synth-stripping after top-up
    for te in te_values:
        input_image = os.path.join(output_path, f'{s}_b0_AP_PA_tu_te{te}.nii.gz')
        output_image = os.path.join(output_path, f'{s}_b0_AP_PA_tu_B0_te{te}.nii.gz')
        if not os.path.exists(os.path.join(output_image)):
            call  = [ f'fslroi', \
                    f'{input_image}', \
                    f'{output_image}', \
                    f'0 -1 0 -1 0 -1 0 1']
            os.system(' '.join(call))

        mri_synthstrip_cmd = [
            'bash', '-c',
            f'source {freesurfer_home}/SetUpFreeSurfer.sh && {mri_synthstrip_path} '
            f'--image {os.path.join(output_path, f"{s}_b0_AP_PA_tu_B0_te{te}.nii.gz")} '
            f'--out {os.path.join(output_path, f"{s}_b0_AP_PA_tu_B0_brain_te{te}.nii.gz")} '
            f'--mask {os.path.join(output_path, f"{s}_b0_AP_PA_tu_B0_brainmask_te{te}.nii.gz")}',
        ]
        process = subprocess.Popen(mri_synthstrip_cmd, env=env)
        process.wait()  # Wait for the process to complete.

    #NOTE : you can run this in case the mask needed to be dilated
    for te in te_values:
        input_image = os.path.join(output_path, f'{s}_b0_AP_PA_tu_B0_brainmask_te{te}.nii.gz')
        output_image = os.path.join(output_path, f'{s}_b0_AP_PA_tu_B0_brainmask_dil_te{te}.nii.gz')
        if not os.path.exists(os.path.join(output_image)):
            call  = [ f'fslmaths', \
                    f'{input_image}', \
                    f'-dilM', \
                    f'{output_image}']
            os.system(' '.join(call))

    ##### 4. Eddy distortion and motion correction #####
    for te in te_values:
        dwi  = os.path.join(output_path, f'{s}_nordicdn_dg_te{te}.nii.gz')
        bvals = os.path.join(input_path, f'{s}_ses-1_acq-b5_dir-AP_echo-0{te}_dwi.bval')
        bvecs = os.path.join(input_path, f'{s}_ses-1_acq-b5_dir-AP_echo-0{te}_dwi.bvec')
        mask  = os.path.join(output_path, f'{s}_b0_AP_PA_tu_B0_brainmask_te{te}.nii.gz')
        acqp  = os.path.join(output_path + '/input', 'acqp.txt')
        index = os.path.join(output_path + '/input', 'index.txt')
        top  = os.path.join(output_path, f'{s}_topup_results_te080') # topup results for TE = 80, change if you want to use another TE
        eddy = os.path.join(output_path, f'{s}_nordicdn_dg_tu_ec_te{te}.nii.gz')
        print('5. Eddy correction for TE = ', te)
        if not os.path.exists(eddy):    
            call = [f'eddy_cuda10.2', \
                f'--imain={dwi}', \
                f'--mask={mask}', \
                f'--index={index}', \
                f'--acqp={acqp}', \
                f'--bvecs={bvecs}', \
                f'--bvals={bvals}', \
                f'--topup={top}', \
                f'--out={eddy}',\
                f'--data_is_shelled']
            os.system(' '.join(call))
        else: 
            print(f'eddy_cuda10.2 has already been run for echo time: {te}')


                                                ####### Supplementary steps for Motion Correction and Axon Diameter input ########

    # make a new folder called preprocessed in the output_path but with the sub-ctrl name.
    preprocessed_path = os.path.join(output_path, f'{s}_preprocessed')
    os.makedirs(preprocessed_path, exist_ok=True)
    # copy all eddy corrected images and NORDIC_denoise images bval and bvec  to this folder. 
    for te in te_values:
        eddy = os.path.join(output_path, f'{s}_nordicdn_dg_tu_ec_te{te}.nii.gz')
        preprocessed = os.path.join(preprocessed_path, f'{s}_nordicdn_dg_tu_ec_te{te}.nii.gz')
        os.system(f'cp {eddy} {preprocessed}')
        nordicdn = os.path.join(output_path, f'{s}_nordicdn_te{te}.nii')
        preprocessed = os.path.join(preprocessed_path, f'{s}_nordicdn_te{te}.nii')
        os.system(f'cp {nordicdn} {preprocessed}')
        os.system(f'cp {input_path}/{s}_ses-1_acq-b5_dir-AP_echo-0{te}_dwi.bval {preprocessed_path}')
        os.system(f'cp {input_path}/{s}_ses-1_acq-b5_dir-AP_echo-0{te}_dwi.bvec {preprocessed_path}')
        
        print(f'copied {eddy}, {nordicdn} and the bval and bvec to {preprocessed_path}')
        # ------------------------ Define data for motion correction using predicted DTI signal ----------------------------------------
        dataTE = os.path.join(preprocessed_path, f'{s}_nordicdn_dg_tu_ec_te{te}.nii.gz')
        bvec = os.path.join(preprocessed_path, f'{s}_ses-1_acq-b5_dir-AP_echo-0{te}_dwi.bvec')
        bval = os.path.join(preprocessed_path, f'{s}_ses-1_acq-b5_dir-AP_echo-0{te}_dwi.bval')
        path2axond = os.path.join(main_path, 'processing', f'{s}_axond')
        os.makedirs(path2axond, exist_ok=True)
        Data_folder = os.path.join(preprocessed_path, f'DataTE_{te}') 

        # --------------- MOTION CORRECTION after Eddy but with ANTs--------------------------------------------
        # First step: Compute DTI

        os.system('mkdir ' + Data_folder)

        in_data = dataTE
        out_data = 'Predicted_' + dataTE
        os.system(f'cp {output_path}/{s}_b0_AP_PA_tu_B0_brainmask_te{te}.nii.gz {Data_folder}/mask.nii.gz')
        os.system('dtifit -k ' + in_data + ' -o ' + os.path.join(Data_folder, 'recon') + ' -m ' + os.path.join(Data_folder, 'mask') + ' -r ' + bvec + ' -b ' + bval)

        print('Computing the diffusion signal')
        Predicted_Signal = estimate_signal_from_DTI(Data_folder, bval, bvec)

        Signal = nib.load(in_data)
        outImg = nib.Nifti1Image(Predicted_Signal, Signal.affine)
        nib.save(outImg, os.path.join(Data_folder, 'Signal_dti.nii.gz'))

        os.system('cp ' + in_data + ' ' + os.path.join(Data_folder, 'Signal.nii.gz'))

        os.system('fslsplit ' + os.path.join(Data_folder, 'Signal_dti.nii.gz') + ' ' + os.path.join(Data_folder, 'Sdti') + ' -t')
        os.system('fslsplit ' + os.path.join(Data_folder, 'Signal.nii.gz') + ' ' + os.path.join(Data_folder, 'S') + ' -t')

        Niters = 2
        for i in range(Niters):
            print('Motion correction using ANTs, iter:', i + 1)
            for vol in range(Signal.shape[3]):
                if vol < 10:
                    number = '000' + str(vol)
                elif (vol >= 10) and (vol < 100):
                    number = '00' + str(vol)
                elif (vol >= 100):
                    number = '0' + str(vol)
                # end if
                img_S = nib.load(os.path.join(Data_folder, 'S' + number + '.nii.gz'))
                fm = ants.image_read(os.path.join(Data_folder, 'S' + number + '.nii.gz'))

                img_Sdti = nib.load(os.path.join(Data_folder, 'Sdti' + number + '.nii.gz'))
                fi = ants.image_read(os.path.join(Data_folder, 'Sdti' + number + '.nii.gz'))

                mask_input = ants.image_read(os.path.join(Data_folder, 'mask.nii.gz'))

                reg = ants.registration(fixed=fi, moving=fm, type_of_transform='Rigid', mask=mask_input, verbose=False, interpolator='linear')
                motion_corrected = ants.apply_transforms(fixed=fi, moving=fm, transformlist=reg['fwdtransforms'], interpolator='linear')
                ants.image_write(motion_corrected, os.path.join(Data_folder, 'Scorr_' + number + '.nii.gz'))
            # end vol
            os.system('fslmerge -t ' + os.path.join(Data_folder, 'Signal_corr') + ' ' + os.path.join(Data_folder, 'Scorr_**'))

            # Recomputing the diffusion tensor again from corrected data
            os.system('dtifit -k ' + os.path.join(Data_folder, 'Signal_corr') + ' -o ' + os.path.join(Data_folder, 'recon') + ' -m ' + os.path.join(Data_folder, 'mask') + ' -r ' + bvec + ' -b ' + bval)

            print('Re-computing the diffusion signal')
            Predicted_Signal = estimate_signal_from_DTI(Data_folder, bval, bvec)
            outImg = nib.Nifti1Image(Predicted_Signal, Signal.affine)
            nib.save(outImg, os.path.join(Data_folder, 'Signal_dti.nii.gz'))

            os.system('rm ' + os.path.join(Data_folder, 'Sdti**'))

            os.system('fslsplit ' + os.path.join(Data_folder, 'Signal_dti.nii.gz') + ' ' + os.path.join(Data_folder, 'Sdti') + ' -t')

            # checking the residuals
            os.system('fslmaths ' + os.path.join(Data_folder, 'Signal') + ' -sub ' + os.path.join(Data_folder, 'Signal_corr.nii.gz') + ' ' + os.path.join(Data_folder, 'res'))
        # end iter
        os.system('rm ' + os.path.join(Data_folder, 'Sdti**'))
        os.system('rm ' + os.path.join(Data_folder, 'Scorr_**'))
        os.system('rm ' + os.path.join(Data_folder, 'S00**'))
        os.system('cp ' + os.path.join(Data_folder, 'Signal_corr.nii.gz') + ' ' + os.path.join(preprocessed_path, f'{s}_Signal_TE{te}_corr.nii.gz'))

        Signal = nib.load(os.path.join(preprocessed_path, f'{s}_Signal_TE{te}_corr.nii.gz'))
        Data = Signal.get_fdata()
        b = np.loadtxt(bval)
        SM = np.mean(Data[:, :, :, b > 0], axis=3)
        outImg = nib.Nifti1Image(SM, Signal.affine)
        nib.save(outImg, os.path.join(preprocessed_path, f'{s}_SMT_TE{te}.nii.gz'))
    # Merge the SMT images into a 4D dataset and save it in the path2axond directory
    merged_output = os.path.join(path2axond, f'{s}_Data4D_new.nii.gz')
    os.system(f'fslmerge -t {merged_output} {os.path.join(preprocessed_path, f"{s}_SMT_TE080.nii.gz")} {os.path.join(preprocessed_path, f"{s}_SMT_TE100.nii.gz")} {os.path.join(preprocessed_path, f"{s}_SMT_TE120.nii.gz")} {os.path.join(preprocessed_path, f"{s}_SMT_TE135.nii.gz")}')

    print('       ---------------------------------------------------------------------------            ')
    print('----------------------------------------------------------------------------------------------')
    print('Done! - you can find the preprocessed image and the spherical mean data in the preprocessed folder. The 4D dataset is saved in the /axond directory.')



