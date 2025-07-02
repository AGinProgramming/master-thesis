% filename = 'G:\NKI dataset\Data_nifti\MRI002\ROIs\MRI002_ROIs_tumour_eroded.nii.gz';
% 
% niiData = niftiread(filename);
% 
% disp(size(niiData));
% 
% img = 
% 
% print(niiData)



% load('Z:\GENIM\NKI dataset\Data_nifti\logic_adjusted_masks\adjusted_patient_MRI001\adjusted_prostate_slice_MRI001\adjusted_prostate_slice_15.mat')
% V = niftiread("Z:\GENIM\NKI dataset\Data_nifti\MRI001\NIFTIs\T2.nii")
% 
% imagesc(V(:,:,15)'); hold on; contour(logic_image,'w');colormap('gray')


gunzip('G:\NKI dataset\Data_nifti\MRI002\ROIs\MRI002_ROIs_tumour_eroded.nii.gz');
nii_data = niftiread('G:\NKI dataset\Data_nifti\MRI002\ROIs\MRI002_ROIs_tumour_eroded.nii');
nii_info = niftiinfo('G:\NKI dataset\Data_nifti\MRI002\ROIs\MRI002_ROIs_tumour_eroded.nii');

% disp(nii_data)
% disp(nii_info)

slice_15 = nii_data(:, :, 10);

figure;
imshow(slice_15, []);
title('Slice 15 of MRI002_ROIs_tumour_eroded');