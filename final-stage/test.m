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

roi_data_path = "C:\Users\alexg\Desktop\study\graduation project\code\postate_ROI_output";


patient = T{28,'PatientName'};
slice_number = floor(T{28,'SliceNo_'});
image_path = fullfile(data_path, patient, "NIFTIs", "T2.nii");

gz_files_path_str = strjoin(fullfile('G:\NKI dataset\Data_nifti\', patient, 'ROIs', '*.nii.gz'), filesep);
gz_files = dir(gz_files_path_str);
if isempty(gz_files)
    error('No .nii.gz files found in the directory.');
end

name_file_gz = fullfile(gz_files.folder, gz_files.name);
gunzip(name_file_gz);

nii_files_path_str = strjoin(fullfile('G:\NKI dataset\Data_nifti\', patient, 'ROIs', '*.nii'), filesep);
nii_files = dir(nii_files_path_str);
name_file_nii = fullfile(nii_files.folder, nii_files.name);
nii_data = niftiread(name_file_nii);


slice = nii_data(:, :, slice_number);
slice = rot90(slice, -1);

imagesc(slice);  % 显示第9层
colormap(gray);  % 使用灰度颜色映射
colorbar;  % 显示颜色条
%%
V = niftiread(image_path);
bg = V(:,:,slice_number);
bg = rot90(bg, -1);

imagesc(bg);  % 显示第9层
colormap(gray);  % 使用灰度颜色映射
colorbar;  % 显示颜色条
%%
dim_data = size(bg);
dim_tumor = size(slice);

if dim_data(1) == dim_tumor(1) && dim_data(2) == dim_tumor(2)
    tumor = slice;
else
    tumor = imresize(slice, [dim_data(1) dim_data(2)],'nearest');
end

% minX = coordinates(1, 1) - 20;
% minY = coordinates(1, 2) - 25;
% maxX = coordinates(1, 3) + 20;
% maxY = coordinates(1, 4) + 10;

minX = coordinates(28, 1) - 20;
minY = coordinates(28, 2) - 25;
maxX = coordinates(28, 3) + 20;
maxY = coordinates(28, 4) + 10;


cropped_tumor = tumor(minY:maxY, minX:maxX);

imagesc(cropped_tumor);  % 显示第9层
colormap(gray);  % 使用灰度颜色映射
colorbar;  % 显示颜色条

[rows, cols] = find(cropped_tumor == 1);

coordinates_all{7} = [rows, cols];

%% 14 28