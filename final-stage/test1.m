excel_path = "G:\Annekoos Schaap - code\Data\All_imaging_features_NKI.xlsx";
T = readtable(excel_path, 'Sheet', 'in');


base_path = "G:\NKI dataset\Data_nifti";

all_patients_data = [];

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
    
    % access coordinates
    [borderY, borderX] = find(logic_image);
    minX = min(borderX);
    maxX = max(borderX);
    minY = min(borderY);
    maxY = max(borderY);
    
    % store all patient data
    current_data = [minX, minY, maxX, maxY];
    all_patients_data = [all_patients_data; current_data];
end

% create excel file to store data above
filename = 'coordinates_1.xlsx';
writematrix(all_patients_data, filename);

