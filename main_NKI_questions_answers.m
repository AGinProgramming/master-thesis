clc
close all
clear all
load PatStats_NKI %% what is this? - it is a .mat file with a table that contains details about the pixel dimension, etc. It is in the same folder as the original main_par_for_NKI.m file
% in Z:\GENIM\Annekoos Schaap - code\Radiomics

patient_list = readtable('\\tsclient\Z\dMRI dataset\Patient_list_dMRI3.csv'); %% haven't used this table before. patient numbers seem different also. 
% You're correct, what we need is the equivalent table but for the NKI dataset. If you look at the table 
% you can see that it has information on the patient ID, file name, and the
% names of the images of T2, ADC, the DCE_rescliced, etc. In this code, all
% it needs is the name of the folder or patient IDs. In your case MRI002,
% etc. So it is easy to recreate that for your purpose. 

image_type = 'T2'; 
data_path = '\\tsclient\Z\dMRI dataset\Data_nifti'; %% seem different from the source earlier. should be changed? 
% Yes, please change to the correct directory. Somehow this file has the paths to the other database and not for the NKI one.
% however, when doing your superpixel analysis what you did was to change
% the transpose (rotate, flip, etc) the original image to fit the prostate
% mask (instead of doing the opposite) so before you proceed you need to
% make sure that the superpixel masks that you now created are aligned with
% the nifti images you will be opening.

textmatlist = {'Global','GLCM','NGTDM','GLSZM','GLRLM'};
winsize = 15;
types = {'T2' , 'ADC' , 'DCE_wash_in', 'DCE_peak'}; %% my code just went through T2. should also go through the others? 
% Yes, you need to run the feature extraction for ADC, DCE_wash_in and DCE_peak. How did you run the code
% if you didn't have the correct patient directory? In the next for loop
% the code should be looping through the different image types.

for types_n = [1:length(types)]
 tStart = tic;
 image_type = char(types{types_n});
 fprintf('Image type: %s\n', image_type)
if strcmp(image_type, 'T2')
    save_path_top = 'E:\ASchaap\Feature_maps\T2'; % change the directory where to save the maps - please don't overwrite any of the data in the GENIM folder
    target_scale = 0.4; %% what is this target_scale for? also showed in the feature extraction function. necessary to keep it?
elseif strcmp(image_type, 'ADC')
    save_path_top = 'E:\ASchaap\Feature_maps\ADC';
    target_scale = 0.4;
elseif strcmp(image_type, 'DCE_wash_in')
    save_path_top = 'E:\ASchaap\Feature_maps\DCE_wash_in'; 
    target_scale =  0.4;
elseif strcmp(image_type, 'DCE_peak')
    save_path_top = 'E:\ASchaap\Feature_maps\DCE_peak'; 
    target_scale =  0.4;
end

    for tt = 1:size(patient_list,1)
    tic 

    patient_name = patient_list.Patient{tt};
    mri_path = fullfile(data_path , patient_name , 'NIFTIs', [image_type '.nii']);

    deli_path = fullfile('E:\ASchaap\Delineations', [patient_name '.nii.gz']); % here you need to give as input the superpixel segmentations in the correct orientation 
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
                voxel_info.PixelSpacing = PatStats.T2_PixelSpacing(num(k)); %% is this PatStats the same as PatStats_NKI? if not, where does it come from?
                voxel_info.SliceThickness = PatStats.T2_SliceThickness(num(k));
                CNN_slice = PatStats.CNN_slice(num(k)); %% what is this CNN for? and the slice number is the same as our ones earlier in the code?
                [mri, deli] = t2_preprocess(mri_path, deli_path, CNN_slice); %% t2_preprocess was updated to v2?
            end
        elseif strcmp(image_type,'ADC')
            num =  find(PatStats.Patient == patient_name);
            for k = 1:length(num)
                voxel_info.PixelSpacing = PatStats.T2_PixelSpacing(num(k));
                voxel_info.SliceThickness = PatStats.T2_PixelSpacing(num(k));
                CNN_slice = PatStats.CNN_slice(num(k));
                [mri, deli] = adc_preprocess(mri_path, deli_path); %% adc_preprocess was updated to v2?
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
                voxel_info.PixelSpacing = PatStats.T2_PixelSpacing(num(k));
                voxel_info.SliceThickness = PatStats.T2_PixelSpacing(num(k));
                CNN_slice = PatStats.CNN_slice(num);
                [mri, deli] = t2_preprocess(mri_path, deli_path, CNN_slice(k));
            end
        end
        %% all the mri in [mri, deli] was a 3D matrix. then loop went through all the slices. But it is not necessary for us right?
        %% Because we have already had the exact slice for each patient. 
        % - Yes, in principle we only need to extract the features for the specific slice we
        % are working with. In the loop below you can see that if the
        % delineation / segmentation file is all 0's for a slice that slice
        % will be skipped (check line 106), so in fact it only analyses the
        % slice with data. Your roi_file needs to be also a 3D matrix and
        % not just 2D, the remaining slices will be zeros.

        voxel_info_PixelSpacing = voxel_info.PixelSpacing;
        roi_path = ['E:\ASchaap\Hubert Blach 2\NKI_feature_extraction\myROIs\' patient_name '.nii.gz'];
        if exist(roi_path, 'file')
        roi_file = niftiread(roi_path );
            for kk = [1:size(mri,3)] % look if you could change that %% no need for checking every slice, right? A: The line after will make sure the code only runs for slices with superpixel delineation.
                if mean(roi_file(:,:,kk),"all")> 0 % here the code looks if the segmentation mask has any positive values for the specific slice kk, if not then it proceeds without analyising it

                    if voxel_info_PixelSpacing ~= target_scale
                        a = voxel_info_PixelSpacing/ target_scale;
                        mri_slice = imresize(mri(:,:,kk),[round(double(size(mri(:,:,kk),1))*a),round(double(size(mri(:,:,kk),2))*a)],'cubic','Antialiasing',true);
                    else 
                        mri_slice = mri(:,:,kk);
                    end
                    TextFeature = texture_extraction_movwin_v4( ...
                                                    [patient_name ' SLICE: ' num2str(kk)], ...
                                                    mri_slice, ...
                                                    textmatlist, ...
                                                    winsize, ...
                                                    voxel_info, ...
                                                    target_scale);
                    %% first add labels list outside the loop
                    %% change into texture_extraction_SLIC(fullfilename, image, textmatlist, voxel_info, labels)
                    %% if the textmatlist and voxel_info are the same? (double-check in case different)
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