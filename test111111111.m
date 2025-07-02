% excel_path = "G:\Annekoos Schaap - code\Data\All_imaging_features_NKI.xlsx";
% T = readtable(excel_path, 'Sheet', 'in');
% 
% 
% base_path = "G:\NKI dataset\Data_nifti";
% 
% all_patients_data = [];
% 
% for i = 1:height(T)
%     patient = T{i,'PatientName'};
%     slice_number = floor(T{i,'SliceNo_'});
%     nifti_path = fullfile(base_path, patient, "NIFTIs", "T2.nii");
%     
%     % load NIFTI data and slices
%     V = niftiread(nifti_path);
%     slice = V(:,:,slice_number);
%     
%     % rotation
% %     slice = rot90(slice, 3);
% % 
% %     % Rescale the image slice
% %     slice = double(slice); % Convert to double for computation
% %     minVal = min(slice(:));
% %     maxVal = max(slice(:));
% %     rescaled_slice = (slice - minVal) / (maxVal - minVal) * 255;
% %     rescaled_slice = uint8(rescaled_slice); % Convert back to uint8
% % 
% %     % store rescaled_image for each patient
% %     rescaled_image_path = fullfile("C:\Users\alexg\Desktop\study\graduation project", "rescaled_images", [patient "_slice_" num2str(slice_number) ".mat"]);
% %     % save(rescaled_image_path, "rescaled_slice");
% %     save(char(rescaled_image_path), "rescaled_slice");
%     
% %     % load masks
% %     mask_path = fullfile(base_path, "logic_adjusted_masks", ["adjusted_patient_" patient], ["adjusted_prostate_slice_" patient], ["adjusted_prostate_slice_" num2str(slice_number) ".mat"]);
% %     % If patient is a character vector
% %     mask_path = fullfile(base_path, 'logic_adjusted_masks', strcat('adjusted_patient_', patient), strcat('adjusted_prostate_slice_', patient), strcat('adjusted_prostate_slice_', num2str(slice_number), '.mat'));
%     
% 
%     % If patient is a string
%     mask_path = fullfile(base_path, "logic_adjusted_masks", "adjusted_patient_" + patient, "adjusted_prostate_slice_" + patient, "adjusted_prostate_slice_" + num2str(slice_number) + ".mat");
%     logic_image = load(mask_path).logic_image;
% 
%     
%     dim_data = size(slice);
%     dim_mask = size(logic_image); % same dimensions
% 
%     if dim_data(1) == dim_mask(1) && dim_data(2) == dim_mask(2)
%         prostate_ROI = logic_image;
%     else
%         prostate_ROI = imresize(logic_image, [dim_data(1) dim_data(2)],'nearest');
%     end
%     
%     % access coordinates
%     [borderY, borderX] = find(prostate_ROI);
%     minX = min(borderX);
%     maxX = max(borderX);
%     minY = min(borderY);
%     maxY = max(borderY);
%     
%     % store all patient data
%     current_data = [minX, minY, maxX, maxY];
%     all_patients_data = [all_patients_data; current_data];
% end

clc;clear;close all;

mask = load(['G:\NKI dataset\Data_nifti\logic_adjusted_masks\adjusted_patient_MRI002...' ...
    '\adjusted_prostate_slice_MRI002\adjusted_prostate_slice_10.mat']);
V = niftiread("G:\NKI dataset\Data_nifti\MRI002\NIFTIs\T2.nii");

% imagesc(V(:,:,10)'); 
% hold on; 
% contour(logic_image,'w');
% colormap('gray')


dim_data = size(V);
dim_mask = size(mask.logic_image);

if dim_data(1) == dim_mask(1) && dim_data(2) == dim_mask(2)

    prostate_ROI = mask.logic_image;
    figure();imagesc(V(:,:,10)'); colormap('gray');
    hold on;
    contour(prostate_ROI,'w')
    title(strcat('Prostate contour overlaid on T2w - '))
    set(gca,'XTick',[])
    set(gca,'YTick',[])
    % access coordinates
    [borderY, borderX] = find(prostate_ROI);
    minX = min(borderX)
    maxX = max(borderX)
    minY = min(borderY)
    maxY = max(borderY)

%     % 在图中画线表示 minX 和 maxX
%     line([minX minX], [1 size(V, 1)], 'Color', 'r', 'LineWidth', 2); % 红色线表示 minX
%     line([maxX maxX], [1 size(V, 1)], 'Color', 'r', 'LineWidth', 2); % 红色线表示 maxX
%     
%     % 在图中画线表示 minY 和 maxY
%     line([1 size(V, 2)], [minY minY], 'Color', 'b', 'LineWidth', 2); % 蓝色线表示 minY
%     line([1 size(V, 2)], [maxY maxY], 'Color', 'b', 'LineWidth', 2); % 蓝色线表示 maxY


else
    prostate_ROI = imresize(mask.logic_image, [dim_data(1) dim_data(2)],'nearest');
    figure();imagesc(V(:,:,10)'); colormap('gray');
    hold on;
    contour(prostate_ROI,'w')
    title(strcat('Prostate contour overlaid on T2w - '))
    set(gca,'XTick',[])
    set(gca,'YTick',[])
end


    








