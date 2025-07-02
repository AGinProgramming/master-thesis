load("G:\NKI dataset\Data_nifti\logic_adjusted_masks\adjusted_patient_MRI056\adjusted_prostate_slice_MRI056\adjusted_prostate_slice_10.mat");
V = niftiread("G:\NKI dataset\Data_nifti\MRI056\NIFTIs\T2.nii");

imagesc(V(:,:,10)'); 
hold on; 
contour(logic_image,'w');
colormap('gray')
