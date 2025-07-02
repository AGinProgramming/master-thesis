excel_path = "G:\Annekoos Schaap - code\Data\All_imaging_features_NKI.xlsx";
T = readtable(excel_path, 'Sheet', 'in');


base_path = "G:\NKI dataset\Data_nifti";

all_patients_data = [];
%%
for i = 1:height(T)
    patient = T{i,'PatientName'};
    slice_number = floor(T{i,'SliceNo_'});
    nifti_path = fullfile(base_path, patient, "NIFTIs", "T2.nii");
    
    % load NIFTI data and slices
    V = niftiread(nifti_path);
    slice = V(:,:,slice_number);
    
    % rotation
%     slice = rot90(slice, 3);
% 
%     % Rescale the image slice
%     slice = double(slice); % Convert to double for computation
%     minVal = min(slice(:));
%     maxVal = max(slice(:));
%     rescaled_slice = (slice - minVal) / (maxVal - minVal) * 255;
%     rescaled_slice = uint8(rescaled_slice); % Convert back to uint8
% 
%     % store rescaled_image for each patient
%     rescaled_image_path = fullfile("C:\Users\alexg\Desktop\study\graduation project", "rescaled_images", [patient "_slice_" num2str(slice_number) ".mat"]);
%     % save(rescaled_image_path, "rescaled_slice");
%     save(char(rescaled_image_path), "rescaled_slice");
    
%     % load masks
%     mask_path = fullfile(base_path, "logic_adjusted_masks", ["adjusted_patient_" patient], ["adjusted_prostate_slice_" patient], ["adjusted_prostate_slice_" num2str(slice_number) ".mat"]);
%     % If patient is a character vector
%     mask_path = fullfile(base_path, 'logic_adjusted_masks', strcat('adjusted_patient_', patient), strcat('adjusted_prostate_slice_', patient), strcat('adjusted_prostate_slice_', num2str(slice_number), '.mat'));
    

    % If patient is a string
    mask_path = fullfile(base_path, "logic_adjusted_masks", "adjusted_patient_" + patient, "adjusted_prostate_slice_" + patient, "adjusted_prostate_slice_" + num2str(slice_number) + ".mat");
    logic_image = load(mask_path).logic_image;

    
    dim_data = size(slice);
    dim_mask = size(logic_image); % same dimensions

    if dim_data(1) == dim_mask(1) && dim_data(2) == dim_mask(2)
        prostate_ROI = logic_image;
    else
        prostate_ROI = imresize(logic_image, [dim_data(1) dim_data(2)],'nearest');
    end
    
    % access coordinates
    [borderY, borderX] = find(prostate_ROI);
    minX = min(borderX);
    maxX = max(borderX);
    minY = min(borderY);
    maxY = max(borderY);
    
    % store all patient data
    current_data = [minX, minY, maxX, maxY];
    all_patients_data = [all_patients_data; current_data];
end

% create excel file to store data above
filename = 'coordinates.xlsx';
writematrix(all_patients_data, filename);










%load("G:\NKI dataset\Data_nifti\logic_adjusted_masks\adjusted_patient_MRI001\adjusted_prostate_slice_MRI001\adjusted_prostate_slice_15.mat");
%V = niftiread("G:\NKI dataset\Data_nifti\MRI001\NIFTIs\T2.nii");

%imagesc(V(:,:,15)'); 
%hold on; 
%contour(logic_image,'w');
%colormap('gray')


%[borderY, borderX] = find(logic_image);


%minX = min(borderX);
%maxX = max(borderX);
%minY = min(borderY);
%maxY = max(borderY);

%fprintf('min border X coordinate: %d, max border X coordinate: %d\n', minX, maxX);
%fprintf('min border Y coordinate: %d, max border Y coordinate: %d\n', minY, maxY);



%% create excel file to store data above';
%data = [minX, minY, maxX, maxY];

%filename = 'coordinates.xlsx'
%writematrix(data, filename);
%%
patient = T{3,'PatientName'};
disp(patient)
slice_number = floor(T{3,'SliceNo_'});
nifti_path = fullfile(base_path, patient, "NIFTIs", "T2.nii");

V = niftiread(nifti_path);
slice = V(:,:,slice_number);
slice = rot90(slice, 3);
slice = fliplr(slice);
mask_path = fullfile(base_path, "logic_adjusted_masks", "adjusted_patient_" + patient, "adjusted_prostate_slice_" + patient, "adjusted_prostate_slice_" + num2str(slice_number) + ".mat");
logic_image = load(mask_path).logic_image;

dim_data = size(slice);
dim_mask = size(logic_image); % same dimensions

if dim_data(1) == dim_mask(1) && dim_data(2) == dim_mask(2)
    prostate_ROI = logic_image;
else
    prostate_ROI = imresize(logic_image, [dim_data(1) dim_data(2)],'nearest');
end

% access coordinates
[borderY, borderX] = find(prostate_ROI);
minX = min(borderX);
maxX = max(borderX);
minY = min(borderY);
maxY = max(borderY);
disp(minX);
disp(minY);
disp(maxX);
disp(maxY);

figure;
imagesc(slice);
colormap('gray');
axis equal;
hold on;

% 获取逻辑图像的边界
boundaries = bwboundaries(prostate_ROI);

% 在原图上绘制逻辑图像的边界
for k = 1:length(boundaries)
    boundary = boundaries{k};
    plot(boundary(:,2), boundary(:,1), 'r', 'LineWidth', 2);
end

% title('Kappa with Superpixel Contours for kk from 1 to 22');
hold off;
