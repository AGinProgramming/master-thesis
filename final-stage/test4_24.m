excel_path = "G:\Annekoos Schaap - code\Data\All_imaging_features_NKI.xlsx";
T = readtable(excel_path, 'Sheet', 'in');


base_path = "G:\NKI dataset\Data_nifti";

all_patients_data = [];
all_pixel_counts = [];

for i = 1:height(T)
    patient = T{i,'PatientName'};
    slice_number = floor(T{i,'SliceNo_'});
    nifti_path = fullfile(base_path, patient, "NIFTIs", "T2.nii");
    
    % load NIFTI data and slices
    V = niftiread(nifti_path);
    slice = V(:,:,slice_number);

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
    
    % 生成图像文件名和保存路径
    patient_1 = patient{1};
    roi_image_name = sprintf('%s_%d_prostate_ROI.png', patient_1, slice_number);
    roi_image_path = fullfile("C:\Users\alexg\Desktop\study\graduation project\code\prostate_ROI_429", roi_image_name);
    
    % 将 ROI 保存为图像文件
    imwrite(prostate_ROI, roi_image_path);

    pixel_count = sum(prostate_ROI(:));
    all_pixel_counts = [all_pixel_counts; pixel_count];


    % access coordinates
    [borderY, borderX] = find(prostate_ROI);
    minX = min(borderX);
    maxX = max(borderX);
    minY = min(borderY);
    maxY = max(borderY);
    
    % store all patient data
    current_data = [minX, minY, maxX, maxY];
    all_patients_data = [all_patients_data; current_data];

    all_data = [all_patients_data, all_pixel_counts];
end

% create excel file to store data above
filename = 'coordinates_429.xlsx';
writematrix(all_data, filename);