clc
close all
clear all
load PatStats_NKI
patient_list = readtable('C:\Users\alexg\Desktop\study\graduation project\code\results5_6.xlsx'); 
image_type = 'T2'; 
data_path = 'G:\NKI dataset\Data_nifti';

textmatlist = {'Global','GLCM','NGTDM','GLSZM','GLRLM'};
types = {'T2' , 'ADC' , 'DCE_wash_in', 'DCE_peak'}; 

coordinate = load('all_patients_data.mat', 'all_patients_data');
coordinates = coordinate.all_patients_data;

path = 'G:\Annekoos Schaap - code\Data\All_imaging_features_NKI.xlsx';
T = readtable(path, 'Sheet', 'in');

if height(coordinates) ~= height(T)
    error('The number of rows in coordinates does not match the number of rows in T.');
end

coordinates_all = cell(height(T), 1);

for i = 1:height(T)
    patient = T{i,'PatientName'};
    slice_number = floor(T{i,'SliceNo_'});
    image_path = fullfile(base_path, patient, "NIFTIs", "T2.nii");

    niiGzFile = fullfile('G:\NKI dataset\Data_nifti\', patient, 'ROIs', [patient '_ROIs_tumour_eroded.nii.gz']);
    gunzip(niiGzFile);
    niiFile = fullfile('G:\NKI dataset\Data_nifti\', patient, 'ROIs', [patient '_ROIs_tumour_eroded.nii']);
    nii_data = niftiread(niiFile);

    slice = nii_data(:, :, slice_number);

    V = niftiread(image_path);
    bg = V(:,:,slice_number);

    dim_data = size(bg);
    dim_tumor = size(slice);

    if dim_data(1) == dim_tumor(1) && dim_data(2) == dim_tumor(2)
        tumor = slice;
    else
        tumor = imresize(slice, [dim_data(1) dim_data(2)],'nearest');
    end

    minX = coordinates(i, 1) - 20;
    minY = coordinates(i, 2) - 25;
    maxX = coordinates(i, 3) + 20;
    maxY = coordinates(i, 4) + 10;

    cropped_tumor = tumor(minY:maxY, minX:maxX);

    [rows, cols] = find(tumor == 1);

    coordinates_all{i} = [rows, cols];
end

save('tumor_coordinates.mat', 'coordinates_all');



%% labels = 把label设计为三维矩阵，所有包括tumor区域的superpixels所含的pixels


for types_n = [1:length(types)]
 tStart = tic;
 image_type = char(types{types_n});
 fprintf('Image type: %s\n', image_type)
if strcmp(image_type, 'T2')
    save_path_top = 'C:\Users\alexg\Desktop\study\graduation project\code\feature_maps\T2';
    target_scale = 0.3;
elseif strcmp(image_type, 'ADC')
    save_path_top = 'C:\Users\alexg\Desktop\study\graduation project\code\feature_maps\ADC';
    target_scale = 0.3;
elseif strcmp(image_type, 'DCE_wash_in')
    save_path_top = 'C:\Users\alexg\Desktop\study\graduation project\code\feature_maps\DCE_wash_in'; 
    target_scale =  0.3;
elseif strcmp(image_type, 'DCE_peak')
    save_path_top = 'C:\Users\alexg\Desktop\study\graduation project\code\feature_maps\DCE_peak'; 
    target_scale =  0.3;
end

    for tt = 1:size(patient_list,1)
    tic 

    patient_name = patient_list.Patient_ID{tt};
    mri_path = fullfile(data_path , patient_name , 'NIFTIs', [image_type '.nii']);

    deli_path = fullfile('E:\ASchaap\Delineations', [patient_name '.nii.gz']);
    save_path = fullfile(save_path_top , patient_name);
    if exist(mri_path,'file')
        fprintf('%s\n', patient_name)
        if ~exist(save_path, "dir")
            mkdir(save_path);
        end
        voxel_info = struct();
        if strcmp(image_type, 'T2')
            num =  find(PatStats.Patient == patient_name);
            for k = 1:length(num)
                voxel_info.PixelSpacing = PatStats.T2_PixelSpacing(num(k));
                voxel_info.SliceThickness = PatStats.T2_SliceThickness(num(k));
                CNN_slice = PatStats.CNN_slice(num(k)); 
                [mri, deli] = t2_preprocess_v2(mri_path, deli_path, CNN_slice);dsfbh
            end
        elseif strcmp(image_type,'ADC')
            num =  find(PatStats.Patient == patient_name);
            for k = 1:length(num)
                voxel_info.PixelSpacing = PatStats.T2_PixelSpacing(num(k));
                voxel_info.SliceThickness = PatStats.T2_PixelSpacing(num(k));
                CNN_slice = PatStats.CNN_slice(num(k));
                [mri, deli] = adc_preprocess_v2(mri_path, deli_path);
            end
        elseif strcmp(image_type,'DCE_wash_in')
            num =  find(PatStats.Patient == patient_name);
            for k = 1:length(num)
                voxel_info.PixelSpacing = PatStats.T2_PixelSpacing(num(k));
                voxel_info.SliceThickness = PatStats.T2_PixelSpacing(num(k));
                CNN_slice = PatStats.CNN_slice(num(k));
                [mri, deli] = t2_preprocess_v2(mri_path, deli_path, CNN_slice);
            end
        elseif strcmp(image_type,'DCE_peak')
            num =  find(PatStats.Patient == patient_name);
            for k = 1:length(num)
                voxel_info.PixelSpacing = PatStats.T2_PixelSpacing(num(k));
                voxel_info.SliceThickness = PatStats.T2_PixelSpacing(num(k));
                CNN_slice = PatStats.CNN_slice(num);
                [mri, deli] = t2_preprocess_v2(mri_path, deli_path, CNN_slice(k));
            end
        end

        voxel_info_PixelSpacing = voxel_info.PixelSpacing;
        %% change the roi_path!
        roi_path = ['E:\ASchaap\Hubert Blach 2\NKI_feature_extraction\myROIs\' patient_name '.nii.gz'];
        if exist(roi_path, 'file')
        roi_file = niftiread(roi_path );
            for kk = [1:size(mri,3)]
                if mean(roi_file(:,:,kk),"all")> 0
                    if voxel_info_PixelSpacing ~= target_scale
                        a = voxel_info_PixelSpacing/ target_scale;
                        mri_slice = imresize(mri(:,:,kk),[round(double(size(mri(:,:,kk),1))*a),round(double(size(mri(:,:,kk),2))*a)],'cubic','Antialiasing',true);
                    else 
                        mri_slice = mri(:,:,kk);
                    end

                    TextFeature = texture_extraction_SLIC([patient_name ' SLICE: ' num2str(kk)], ...
                                                      mri_slice, ...
                                                      textmatlist, ...
                                                      voxel_info, ...
                                                      labels(:,:,tt));
%                     TextFeature = texture_extraction_movwin_v4( ...
%                                                     [patient_name ' SLICE: ' num2str(kk)], ...
%                                                     mri_slice, ...
%                                                     textmatlist, ...
%                                                     winsize, ...
%                                                     voxel_info, ...
%                                                     target_scale);
                    if kk<10
                        kk_num = ['0' num2str(kk)];
                    else
                        kk_num = [ num2str(kk)];
                    end
                    try
                        sv = fullfile(save_path, ['Slice_' kk_num '.mat']);
                        parsave(sv,TextFeature);
                         fprintf('Saved %s \n', ['Slice ' kk_num ] )
        
                            
                    catch
                        fprintf('Save error \n')
                    end
                end
            end
        end
    end
    t_el = toc;

    end
  tEnd = toc(tStart);      
  fprintf('***********************************************************************************************\n')
  fprintf('Time elapsed  in sequance %G min \n' , tEnd/60)
  fprintf('***********************************************************************************************\n')
end