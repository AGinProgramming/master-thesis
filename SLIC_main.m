clc
close all
clear all
load PatStats_NKI
patient_list = readtable('\\tsclient\Z\dMRI dataset\Patient_list_dMRI3.csv');

image_type = 'T2'; 
data_path = '\\tsclient\Z\dMRI dataset\Data_nifti'; % 'E:\ASchaap\Hubert Blach 2\NKI_feature_extraction\Data_nifti' ;
textmatlist = {'Global','GLCM','NGTDM','GLSZM','GLRLM'};
winsize = 15;
types = {'T2' , 'ADC' , 'DCE_wash_in', 'DCE_peak'};

for types_n = [1:length(types)]
 tStart = tic;
 image_type = char(types{types_n});
 fprintf('Image type: %s\n', image_type)
if strcmp(image_type, 'T2')
    save_path_top = 'E:\ASchaap\Feature_maps\T2'; % 'E:\ASchaap\Hubert Blach 2\NKI_feature_extraction\featureMapsROIs3\T2';
    target_scale = 0.4;
elseif strcmp(image_type, 'ADC')
    save_path_top = 'E:\ASchaap\Feature_maps\ADC'; % 'E:\ASchaap\Hubert Blach 2\NKI_feature_extraction\featureMapsROIs3\ADC_v2';
    target_scale = 0.4;
elseif strcmp(image_type, 'DCE_wash_in')
    save_path_top = 'E:\ASchaap\Feature_maps\DCE_wash_in'; % 'E:\ASchaap\Hubert Blach 2\NKI_feature_extraction\featureMapsROIs3\DCE_wash_in';
    target_scale =  0.4;
elseif strcmp(image_type, 'DCE_peak')
    save_path_top = 'E:\ASchaap\Feature_maps\DCE_peak'; % 'E:\ASchaap\Hubert Blach 2\NKI_feature_extraction\featureMapsROIs3\DCE_peak';
    target_scale =  0.4;
end
%     for tt = 1:59 % Missing patients: 1, 3, 48, 51, 52
    for tt = 1:size(patient_list,1)
    tic 
%     if tt<10
%         patient_name = ['MRI00' num2str(tt)];
%     else 
%         patient_name = ['MRI0' num2str(tt)];
%     end
    patient_name = patient_list.Patient{tt};
    mri_path = fullfile(data_path , patient_name , 'NIFTIs', [image_type '.nii']);
%     deli_path = fullfile('E:\ASchaap\Hubert Blach 2\NKID_New_data\CNN nii', [patient_name '.nii']);
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
                [mri, deli] = t2_preprocess(mri_path, deli_path, CNN_slice);
            end
        elseif strcmp(image_type,'ADC')
            num =  find(PatStats.Patient == patient_name);
            for k = 1:length(num)
                voxel_info.PixelSpacing = PatStats.T2_PixelSpacing(num(k));
                voxel_info.SliceThickness = PatStats.T2_PixelSpacing(num(k));
                CNN_slice = PatStats.CNN_slice(num(k));
                [mri, deli] = adc_preprocess(mri_path, deli_path);
            end
        elseif strcmp(image_type,'DCE_wash_in')
            num =  find(PatStats.Patient == patient_name);
            for k = 1:length(num)
                voxel_info.PixelSpacing = PatStats.T2_PixelSpacing(num(k));
                voxel_info.SliceThickness = PatStats.T2_PixelSpacing(num(k));
                CNN_slice = PatStats.CNN_slice(num(k));
                [mri, deli] = t2_preprocess(mri_path, deli_path, CNN_slice);
            end
        elseif strcmp(image_type,'DCE_peak')
            num =  find(PatStats.Patient == patient_name);
            for k = 1:length(num)
                voxel_info.PixelSpacing = -.T2_PixelSpacing(num(k));
                voxel_info.SliceThickness = PatStats.T2_PixelSpacing(num(k));
                CNN_slice = PatStats.CNN_slice(num);
                [mri, deli] = t2_preprocess(mri_path, deli_path, CNN_slice(k));
            end
        end
        voxel_info_PixelSpacing = voxel_info.PixelSpacing;
        roi_path = ['E:\ASchaap\Hubert Blach 2\NKI_feature_extraction\myROIs\' patient_name '.nii.gz'];
        if exist(roi_path, 'file')
        roi_file = niftiread(roi_path );
            for kk = [1:size(mri,3)] % look if you could change that 
                if mean(roi_file(:,:,kk),"all")> 0
                    if voxel_info_PixelSpacing ~= target_scale
                        a = voxel_info_PixelSpacing/ target_scale;
                        mri_slice = imresize(mri(:,:,kk),[round(double(size(mri(:,:,kk),1))*a),round(double(size(mri(:,:,kk),2))*a)],'cubic','Antialiasing',true);
                    else 
                        mri_slice = mri(:,:,kk);
                    end
                    TextFeature = texture_extraction_SLIC( ...
                                                    [patient_name ' SLICE: ' num2str(kk)], ...
                                                    mri_slice, ...
                                                    textmatlist, ...
                                                    winsize, ...
                                                    voxel_info, ...
                                                    target_scale);
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
%     fprintf('Time elapsed %G min \n' , t_el/60)
    end
  tEnd = toc(tStart);      % pair 2: toc 
  fprintf('***********************************************************************************************\n')
  fprintf('Time elapsed  in sequance %G min \n' , tEnd/60)
  fprintf('***********************************************************************************************\n')
end