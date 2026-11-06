function sigma_map (dwi_filename, denoised_dwi_filename, output_filename)
    % Load original DWI and denoised DWI
    dwi = load_untouch_nii(dwi_filename);
    dwi_dn = load_untouch_nii(denoised_dwi_filename);
    
    % Compute residuals
    residuals = double(dwi.img) - double(dwi_dn.img);
    
    % Compute standard deviation (sigma) across volumes
    sigma = std(residuals, [], 4);
    
    % Prepare NIfTI structure for sigma map
    nii_struct = dwi;
    nii_struct.hdr.dime.dim(1) = 3;  % Set dimensionality to 3
    nii_struct.hdr.dime.dim(5) = 1;  % Set number of volumes to 1
    nii_struct.hdr.dime.datatype = 64;  % Set datatype to double
    
    % Save sigma map to NIfTI file
    nii_struct.img = double(sigma);  % Convert to single precision to match typical DWI data
    save_untouch_nii(nii_struct, output_filename);
end

