
clear all

listPats = ['MRI002';'MRI045';'MRI056'];
sliceloc = [10,20,10];

for j = 1:size(listPats,1)

    MRI_pat = listPats(j,:);

    % load nifti
    data = niftiread(strcat('G:\NKI dataset\Data_nifti\',MRI_pat,'\NIFTIs\T2.nii'));

    % load mask
    slice = sliceloc(j); % you should retrieve this from the excel file, for now I made a small list
    mask_mat = load(strcat("G:\NKI dataset\Data_nifti\logic_adjusted_masks\adjusted_patient_",MRI_pat,"\adjusted_prostate_slice_",MRI_pat,"\adjusted_prostate_slice_",num2str(slice),".mat"));

    dim_data = size(data);
    dim_mask = size(mask_mat.logic_image); % same dimensions

    if dim_data(1) == dim_mask(1) && dim_data(2) == dim_mask(2)
        prostate_ROI = mask_mat.logic_image;
        figure();imagesc(data(:,:,slice)'); colormap('gray');
        hold on;
        contour(prostate_ROI,'w')
        title(strcat('Prostate contour overlaid on T2w - ',MRI_pat))
        set(gca,'XTick',[])
        set(gca,'YTick',[])

    else
        prostate_ROI = imresize(mask_mat.logic_image, [dim_data(1) dim_data(2)],'nearest');
        figure();imagesc(data(:,:,slice)'); colormap('gray');
        hold on;
        contour(prostate_ROI,'w')
        title(strcat('Prostate contour overlaid on T2w - ',MRI_pat))
        set(gca,'XTick',[])
        set(gca,'YTick',[])
    end
end