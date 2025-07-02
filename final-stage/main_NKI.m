clc
close all
clear all
load PatStats_NKI
% patient_list = readtable('C:\Users\alexg\Desktop\study\graduation project\code\results5_6.xlsx'); 
patient_list = readtable('C:\Users\alexg\Desktop\study\graduation project\code\results86.xlsx'); 
% image_type = 'T2'; 
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




%%

name = T{4,'PatientName'};
sl_n = floor(T{4,'SliceNo_'});
ip_t2 = fullfile(data_path, name, "NIFTIs", "T2.nii");
ip_adc = fullfile(data_path, name, "NIFTIs", "ADC.nii");
ip_dcep = fullfile(data_path, name, "NIFTIs", "DCE_peak.nii");
ip_decw = fullfile(data_path, name, "NIFTIs", "DCE_wash_in.nii");
image_paths = {ip_t2, ip_adc, ip_dcep, ip_decw};

% V = niftiread(image_path);
% bg = V(:,:,slice_number);
% bg = rot90(bg, -1);
% bg = fliplr(bg);

mask_path = fullfile(data_path, "logic_adjusted_masks", "adjusted_patient_" + name, "adjusted_prostate_slice_" + name, "adjusted_prostate_slice_" + num2str(sl_n) + ".mat");
logic_image = load(mask_path).logic_image;

% 创建一个新图窗口
figure;

% 循环读取图像，标记mask并显示
for i = 1:4
    % 读取nii图像
    V = niftiread(image_paths{i});
    
    % 获取指定切片
    bg = V(:,:,sl_n); 
    bg = rot90(bg, -1);
    bg = fliplr(bg);

    % 在图像上叠加mask的轮廓
    subplot(1, 4, i);
    imshow(bg, []);
    hold on;
    contour(logic_image, [0.5 0.5], 'r', 'LineWidth', 2);
    hold off;
    
    % 添加标题
    switch i
        case 1
            title('a) T2', 'Position', [size(bg,2)/2 -size(bg,1)/10], 'VerticalAlignment', 'top');
        case 2
            title('b) ADC', 'Position', [size(bg,2)/2 -size(bg,1)/10], 'VerticalAlignment', 'top');
        case 3
            title('c) DCE Peak', 'Position', [size(bg,2)/2 -size(bg,1)/10], 'VerticalAlignment', 'top');
        case 4
            title('d) DCE Wash-in', 'Position', [size(bg,2)/2 -size(bg,1)/10], 'VerticalAlignment', 'top');
    end
end

% 调整布局，使图像在一行四列中显示
set(gcf, 'Position', [100, 100, 1600, 400]); % 使图像宽度适应显示


%% 在这里出现了问题，查看为什么没有坐标作为tumor的区域，可能是切割有问题

output_folder = 'D:\tumor_area';  % 定义输出文件夹路径
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

newcoordinates = [];

for i = 1:height(T)
    patient = T{i,'PatientName'};
    slice_number = floor(T{i,'SliceNo_'});
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
    slice = fliplr(slice);
%     figure;
%     imshow(slice, []);
%     colormap('gray');
%     axis equal;

%     path1 = fullfile(output_folder, sprintf('patient_%s_slice.png', patient{1}));
%     imwrite(mat2gray(slice), path1);

    V = niftiread(image_path);
    bg = V(:,:,slice_number);
    bg = rot90(bg, -1);
    bg = fliplr(bg);
%     figure;
%     imshow(bg, []);
%     colormap('gray');
%     axis equal;

    dim_data = size(bg);
    dim_tumor = size(slice);

    if dim_data(1) == dim_tumor(1) && dim_data(2) == dim_tumor(2)
        tumor = slice;
    else
        tumor = imresize(slice, [dim_data(1) dim_data(2)],'nearest');
    end

%     path2 = fullfile(output_folder, sprintf('patient_%s_origin.png', patient{1}));
%     imwrite(mat2gray(tumor), path2);

    minX = coordinates(i, 1) - 20;
    minY = coordinates(i, 2) - 25;
    maxX = coordinates(i, 3) + 20;
    maxY = coordinates(i, 4) + 10;
    newcoordinates = [newcoordinates; minX, maxX, minY, maxY];

    cropped_tumor = tumor(minY:maxY, minX:maxX);
    cropped_image = bg(minY:maxY, minX:maxX);
%     output_image_path = fullfile(output_folder, sprintf('patient_%s_slice_%d.png', patient{1}, slice_number));
%     imwrite(mat2gray(cropped_tumor), output_image_path);

    [rows, cols] = find(cropped_tumor == 1);

    coordinates_all{i} = [rows, cols];


%     boundaries = bwboundaries(cropped_tumor);
%     % 显示原图
%     figure;
%     imshow(cropped_image, []);
%     colormap('gray');
%     axis equal;
%     hold on;
% 
%     % 将肿瘤区域的轮廓绘制在原图上
%     for k = 1:length(boundaries)
%         boundary = boundaries{k};
%         plot(boundary(:, 2), boundary(:, 1), 'r', 'LineWidth', 2);
%     end
%     
%     hold off;
% 
%     output_path = fullfile('D:\tumor_position', [patient{1}, '.png']);
%     saveas(gcf, output_path);  % 保存当前图形窗口为 PNG 格式
%     close(gcf);

end
% 

% save('tumor_coordinates.mat', 'coordinates_all');



%% labels = 把label设计为三维矩阵，所有包括tumor区域的superpixels所含的pixels
% 读取 MAT 文件
filename = 'all_labels.mat';
data = load(filename);

% 列出数据字段
fields = fieldnames(data);
%%
best_alpha_values = patient_list.('BestAlphaValue');
%%
indexs = int32((best_alpha_values - 5) / 0.1 + 1);

% 初始化 labels_index 以存储提取的二维数组
labels_index = cell(1, numel(indexs));

% 遍历索引并提取对应的二维数组
for j = 1 : numel(indexs)
    index = (j - 1) * 20 + indexs(j);  % 计算实际索引
    fieldName = sprintf('image_%d', index);  % 获取对应的字段名
    labels_index{j} = data.(fieldName);  % 提取并存储二维数组
end

% 显示提取的结果
disp(size(labels_index));
disp(labels_index);

% for j = 1 : size(indexs)
%     index = (j - 1) * 20 + indexs(j);
%     labels_index(j) = data(index);

% print(size(labels))
% % 显示每个字段的数据
% for i = 1:numel(fields)
%     field_name = fields{i};
%     disp(['Data for ', field_name, ':']);
%     disp(data.(field_name));
% end
%%
% file_name = 'unique_labels.xlsx';
% file_name = 'unique_labels_76.xlsx';
file_name = 'unique_labels_729.xlsx';
unique_labels = readtable(file_name);

rootDir_MRDI = 'G:\NKI dataset\Feature_maps_NKI\MRDI';
patientFolders_MRDI = dir(fullfile(rootDir_MRDI, 'MRI*'));

% resultsFile = 'C:\Users\alexg\Desktop\study\graduation project\code\results5_6.xlsx';
resultsFile = 'C:\Users\alexg\Desktop\study\graduation project\code\results726.xlsx';
patientData = readtable(resultsFile);
targetPatientIDs = patientData.PatientID;

targetFolders = {};
for i = 1:length(targetPatientIDs)
    for j = 1:length(patientFolders_MRDI)
        folderName = patientFolders_MRDI(j).name;
        if contains(folderName, targetPatientIDs{i})
            targetFolders{end+1} = fullfile(rootDir_MRDI, folderName);
            break; % 找到对应文件夹后跳出内层循环
        end
    end
end

featureMaps_MRDI = cell(length(targetFolders), 1);

requiredFields = {'A', 'kappa', 'kep', 'mu', 't0_est'};
%%

% addpath('Vallieres\radiomics');
for types_n = 1:length(types)
     tStart = tic;
     image_type = char(types{types_n});
     fprintf('Image type: %s\n', image_type)
    if strcmp(image_type, 'T2')
        save_path_top = 'C:\Users\alexg\Desktop\study\graduationproject\code\featuremaps729\T2';
        target_scale = 0.3;
    elseif strcmp(image_type, 'ADC')
        save_path_top = 'C:\Users\alexg\Desktop\study\graduationproject\code\featuremaps729\ADC';
        target_scale = 0.3;
    elseif strcmp(image_type, 'DCE_wash_in')
        save_path_top = 'C:\Users\alexg\Desktop\study\graduationproject\code\featuremaps729\DCEwashin'; 
        target_scale =  0.3;
    elseif strcmp(image_type, 'DCE_peak')
        save_path_top = 'C:\Users\alexg\Desktop\study\graduationproject\code\featuremaps729\DCEpeak'; 
        target_scale =  0.3;
    end

    labels_resize_all = cell(1, 35);
    MRDI_maps = [];

    for tt = 1:size(patient_list,1)
%     for tt = 1:2
    tic
    

    patient_name = T.PatientName{tt};
    mri_path = fullfile(data_path , patient_name , 'NIFTIs', [image_type '.nii']);
    patient_slice_number = num2str(floor(T{tt,'SliceNo_'}));
    deli_path = fullfile('G:\NKI dataset\Data_nifti\logic_adjusted_masks', ['adjusted_patient_' patient_name], ['adjusted_prostate_slice_' patient_name], ['adjusted_prostate_slice_' patient_slice_number '.mat']);
    disp(save_path_top)
    disp(['Processing patient: ', patient_name, ', Image type: ', image_type]);
    disp(['MRI path: ', mri_path]);
    disp(['Deli path: ', deli_path]);

    save_path = fullfile(save_path_top , patient_name);
    disp(save_path)
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
%                 deli_path = fullfile(roi_data_path, ['prostate_ROI_patient_' patient_name '_slice_' CNN_slice '.mat']);
                [mri, deli] = t2_preprocess(mri_path, deli_path, CNN_slice);
            end
        elseif strcmp(image_type,'ADC')
            num =  find(PatStats.Patient == patient_name);
            for k = 1:length(num)
                voxel_info.PixelSpacing = PatStats.T2_PixelSpacing(num(k));
                voxel_info.SliceThickness = PatStats.T2_PixelSpacing(num(k));
                CNN_slice = PatStats.CNN_slice(num(k));
%                 deli_path = fullfile(roi_data_path, ['prostate_ROI_patient_' patient_name '_slice_' CNN_slice '.mat']);
                [mri, deli] = adc_preprocess(mri_path, deli_path, CNN_slice);
            end
        elseif strcmp(image_type,'DCE_wash_in')
            num =  find(PatStats.Patient == patient_name);
            for k = 1:length(num)
                voxel_info.PixelSpacing = PatStats.T2_PixelSpacing(num(k));
                voxel_info.SliceThickness = PatStats.T2_PixelSpacing(num(k));
                CNN_slice = PatStats.CNN_slice(num(k));
%                 deli_path = fullfile(roi_data_path, ['prostate_ROI_patient_' patient_name '_slice_' CNN_slice '.mat']);
                [mri, deli] = t2_preprocess(mri_path, deli_path, CNN_slice);
            end
        elseif strcmp(image_type,'DCE_peak')
            num =  find(PatStats.Patient == patient_name);
            for k = 1:length(num)
                voxel_info.PixelSpacing = PatStats.T2_PixelSpacing(num(k));
                voxel_info.SliceThickness = PatStats.T2_PixelSpacing(num(k));
                CNN_slice = PatStats.CNN_slice(num(k));
%                 deli_path = fullfile(roi_data_path, ['prostate_ROI_patient_' patient_name '_slice_' CNN_slice '.mat']);
                [mri, deli] = t2_preprocess(mri_path, deli_path, CNN_slice);
            end
        end

        voxel_info_PixelSpacing = voxel_info.PixelSpacing;

%%%%%%%%%%%%%%% 现在deli_path的路径已经更改完毕，需要调整函数中读取deli_path的数据类型。



%         roi_path = ['G:\NKI dataset\Data_nifti\logic_adjusted_masks\' patient_name, 'ROIs\', patient_name, '*_ROIs_tumour_eroded.nii.nii.gz'
%             patient_name 'ROIs\' 'MRI002_ROIs_tumour_eroded.nii.nii.gz'];
%         roi_path_str = strjoin(fullfile('G:\NKI dataset\Data_nifti\', patient_name, 'ROIs\', 'patient_name', '*_ROIs_tumour_eroded.nii.nii.gz'), filesep);
        disp(size(mri))
        mri_cut = mri((coordinates(tt, 2) - 25):(coordinates(tt, 4) + 10 - 1), (coordinates(tt, 1) - 20):(coordinates(tt, 3) + 20 - 1),:);
        roi_path_str = fullfile('G:\NKI dataset\Data_nifti\', patient_name, 'ROIs\', [patient_name '_ROIs_tumour_eroded.nii.gz']);
        roi_path_dir = dir(roi_path_str);
        disp(roi_path_dir)
        if isempty(roi_path_dir)
            error('No .nii.gz files found in the directory.');
        end

        roi_file_gz = fullfile(roi_path_dir.folder, roi_path_dir.name);
        gunzip(roi_file_gz)

        %roi_path = strjoin(fullfile('G:\NKI dataset\Data_nifti\', patient_name, 'ROIs\', 'patient_name', '*_ROIs_tumour_eroded.nii.nii'), filesep);
        roi_path = fullfile('G:\NKI dataset\Data_nifti\', patient_name, 'ROIs\', [patient_name '_ROIs_tumour_eroded.nii']);
        roi = dir(roi_path);
        roi_nii = fullfile(roi.folder, roi.name);

        if exist(roi_path, 'file')
%         roi_file = niftiread(roi_path );
           roi_file = niftiread(roi_nii );
            for kk = 1:size(roi_file,3)
                if mean(roi_file(:,:,kk),"all")> 0
                    if voxel_info_PixelSpacing ~= target_scale
                        a = voxel_info_PixelSpacing/ target_scale;
                        mri_slice = imresize(mri_cut,[round(double(size(mri_cut,1))*a),round(double(size(mri_cut,2))*a)],'cubic','Antialiasing',true);
                    else 

                        mri_slice = mri_cut;
                    end

                    [mri_height, mri_width] = size(mri_slice);
                    labels = labels_index{tt};
                    labels = double(labels);
                    labels_resize = imresize(labels, [mri_height, mri_width], 'nearest');
%                     labels_resize_all{tt} = labels_resize;
                    
                    unique_labels_cell = table2cell(unique_labels);
                    disp(['Getting to: exraction']);
                    TextFeature = texture_extraction_SLIC(mri_slice, ...
                                                      textmatlist, ...
                                                      voxel_info, ...
                                                      labels_resize, ...
                                                      unique_labels_cell{tt,2});
                    disp(['exraction finished']);
%                     TextFeature = texture_extraction_SLIC([patient_name ' SLICE: ' num2str(kk)], ...
%                                                       mri_slice, ...
%                                                       textmatlist, ...
%                                                       voxel_info, ...
%                                                       labels_index, ...
% %                                                       labels(:,:,tt), ...
%                                                       unique_labels);
%                     TextFeature = texture_extraction_movwin_v4( ...
%                                                     [patient_name ' SLICE: ' num2str(kk)], ...
%                                                     mri_slice, ...
%                                                     textmatlist, ...
%                                                     winsize, ...
%                                                     voxel_info, ...
%                                                     target_scale);
                    % 计算每个superpixel的median value
                    unique_labels_in_slice = str2num(unique_labels_cell{tt,2});
                    weights_in_slice = str2num(unique_labels_cell{tt,3});
                    median_values = zeros(size(unique_labels_in_slice));
                    weights = zeros(size(unique_labels_in_slice));
                    for l = 1:length(unique_labels_in_slice)
                        label = unique_labels_in_slice(l);
                        mask = labels_resize == label;
                        median_values(l) = median(mri_slice(mask));
                        weights(l) = weights_in_slice(l);
                    end

                    % 计算加权平均值
                    weighted_median_value = sum(median_values .* weights) / sum(weights);
                    fprintf('Patient: %s, Weighted Median Value: %.4f\n', patient_name, weighted_median_value);

                    % 将结果保存到TextFeature结构中
                    if ~isfield(TextFeature, 'tumor_value')
                        TextFeature.tumor_value = [];
                    end
                    TextFeature.tumor_value = weighted_median_value;

                    if kk<10
                        kk_num = ['0' num2str(kk)];
                    else
                        kk_num = [ num2str(kk)];
                    end
                    try
                        sv = fullfile(save_path, ['Slice_' kk_num '.mat']);
                        disp(['Saving to: ', sv]);
                        disp(['Data size: ', num2str(size(TextFeature))]);                       
                        parsave(sv,TextFeature);
                        fprintf('Saved %s \n', ['Slice ' kk_num ] )

                            
                    catch ME
                        fprintf('Save error: %s\n', ME.message); % 显示错误信息
%                         fprintf('Save error \n')
                    end
                end
            end
        end

        folder = targetFolders(tt);
        patientDir_MRDI = folder{1};
        matFiles = dir(fullfile(patientDir_MRDI, '*.mat'));
        unique_labels_in_slice = str2num(unique_labels_cell{tt,2});
        weights_in_slice = str2num(unique_labels_cell{tt,3});
        median_values = zeros(size(unique_labels_in_slice));
        weights = zeros(size(unique_labels_in_slice));
        
        for z = 1:length(matFiles)
            filename_MRDI = fullfile(patientDir_MRDI, matFiles(z).name);
            data_MRDI = load(filename_MRDI);
            featureMatrix_MRDI = [];
            for y = 1:length(requiredFields)
                fieldName = requiredFields{y};
                if isfield(data_MRDI, fieldName)
                    value = data_MRDI.(fieldName);
                    value_cropped = value(newcoordinates(tt, 3):(newcoordinates(tt, 4)- 1), newcoordinates(tt, 1):(newcoordinates(tt, 2) - 1));
                    for l = 1:length(unique_labels_in_slice)
                        label = unique_labels_in_slice(l);
                        mask = labels == label;
                        median_values(l) = nanmedian(value_cropped(mask));
                        weights(l) = weights_in_slice(l);
                    end
                    weighted_median_value(y) = sum(median_values .* weights) / sum(weights);
                    disp('done for a patient')
%                     fprintf('Patient: %s, Weighted Median Value: %.4f\n', patient_name, weighted_median_value);
                end
            end
        end
        MRDI_maps = [MRDI_maps; weighted_median_value];

    end
    t_el = toc;

    end

  output_file_MRDI = 'D:\code备份\7.29\Hesheng Gong - code\feature_map_all_MRDI_729.xlsx';
  writematrix(MRDI_maps, output_file_MRDI);

  tEnd = toc(tStart);      
  fprintf('***********************************************************************************************\n')
  fprintf('Time elapsed  in sequance %G min \n' , tEnd/60)
  fprintf('***********************************************************************************************\n')
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 报错原因：某些（fe：15 in MRI002）superpixel对应的mask中没有非零元素


% %%
% % 假设.nii文件的路径
% filePath = 'G:\NKI dataset\Data_nifti\MRI016\ROIs\MRI016_ROIs_tumour_eroded.nii';
% 
% % 使用niftiread函数读取NIfTI文件
% niiData = niftiread(filePath);
% 
% imagesc(niiData(:, :, 9));  % 显示第9层
% colormap(gray);  % 使用灰度颜色映射
% colorbar;  % 显示颜色条
% 
% 
% %%
% coordinate = load('all_patients_data.mat', 'all_patients_data');
% coordinates = coordinate.all_patients_data;
% 
% image_path = fullfile('G:\NKI dataset\Data_nifti\MRI016\NIFTIs\T2.nii');
% 
% gz_files_path_str = 'G:\NKI dataset\Data_nifti\MRI016\ROIs\MRI016_ROIs_tumour_eroded.nii.gz';
% gz_files = dir(gz_files_path_str);
% if isempty(gz_files)
%     error('No .nii.gz files found in the directory.');
% end
% 
% name_file_gz = fullfile(gz_files.folder, gz_files.name);
% gunzip(name_file_gz);
% 
% nii_files_path_str = 'G:\NKI dataset\Data_nifti\MRI016\ROIs\MRI016_ROIs_tumour_eroded.nii';
% % nii_files = dir(nii_files_path_str);
% % name_file_nii = fullfile(nii_files.folder, nii_files.name);
% % nii_data = niftiread(name_file_nii);
% nii_data = niftiread(nii_files_path_str);
% 
% slice = nii_data(:, :, 9);
% 
% imagesc(slice);
% colormap(gray);
% axis equal;
% axis off;
% 
% V = niftiread(image_path);
% bg = V(:,:,9);
% 
% %%问题出在这里，resize之后出现了问题
% 
% dim_data = size(bg);
% dim_tumor = size(slice);
% 
% if dim_data(1) == dim_tumor(1) && dim_data(2) == dim_tumor(2)
%     tumor = slice;
% else
%     tumor = imresize(slice, [dim_data(1) dim_data(2)],'nearest');
% end
% 
% minX = coordinates(7, 1) - 20;
% minY = coordinates(7, 2) - 25;
% maxX = coordinates(7, 3) + 20;
% maxY = coordinates(7, 4) + 10;
% 
% cropped_tumor = tumor(minY:maxY, minX:maxX);
% 
% % % 二值化图像
% % threshold = 0.5; % 设定一个阈值，这里假设阈值为0.5
% % bw_tumor = cropped_tumor > threshold;
% % 
% % % 显示二值化后的黑白图像
% % imagesc(bw_tumor);
% % colormap(gray);
% % axis equal;
% % axis off;
% 
% [rows, cols] = find(cropped_tumor == 1);
% 
% coordinates_all{1} = [rows, cols];

% %%
% mri_path = fullfile(data_path , 'MRI002\NIFTIs\ADC.nii');
% deli_path = 'G:\NKI dataset\Data_nifti\logic_adjusted_masks\adjusted_patient_MRI002\adjusted_prostate_slice_MRI002\adjusted_prostate_slice_10.mat';
% voxel_info = struct();
% num =  find(PatStats.Patient == patient_name);
% voxel_info.PixelSpacing = PatStats.T2_PixelSpacing(num(1));
% voxel_info.SliceThickness = PatStats.T2_SliceThickness(num(1));
% CNN_slice = PatStats.CNN_slice(num(1));
% [mri1, deli1] = t2_preprocess(mri_path, deli_path, CNN_slice);


% %%
% rootDir_ADC = 'C:\Users\alexg\Desktop\study\graduationproject\code\featuremaps76\ADC\';
% requiredFields = {'grayscale'};
% patientFolders_ADC = dir(fullfile(rootDir_ADC, 'MRI*'));
% featureMaps_ADC = cell(length(patientFolders_ADC), 1);
% patientDir_ADC = fullfile(rootDir_ADC, patientFolders_ADC(31).name);
% matFiles = dir(fullfile(patientDir_ADC, '*.mat'));
% filename = fullfile(patientDir_ADC, matFiles(1).name);
% data = load(filename);
% fields = fieldnames(data.TextFeature);
% structData = data.TextFeature.(requiredFields{1});
% 
% 
% 
% 
% %%
% rootDir_ADC = 'C:\Users\alexg\Desktop\study\graduationproject\code\featuremaps716\ADC\';
% rootDir_DCEpeak = 'C:\Users\alexg\Desktop\study\graduationproject\code\featuremaps716\DCEpeak\';
% rootDir_DCEwashin = 'C:\Users\alexg\Desktop\study\graduationproject\code\featuremaps716\DCEwashin\';
% rootDir_T2 = 'C:\Users\alexg\Desktop\study\graduationproject\code\featuremaps716\T2\';
% 
% requiredFields = {'grayscale'};
% 
% patientFolders_ADC = dir(fullfile(rootDir_ADC, 'MRI*'));
% featureMaps_ADC = cell(length(patientFolders_ADC), 1);
% for i = 1:length(patientFolders_ADC)
%     patientDir_ADC = fullfile(rootDir_ADC, patientFolders_ADC(i).name);
%     
%     % 获取当前patient文件夹下的所有.mat文件
%     matFiles = dir(fullfile(patientDir_ADC, '*.mat'));
%     
%     for j = 1:length(matFiles)
%         % 构建文件路径
%         filename = fullfile(patientDir_ADC, matFiles(j).name);
%         
%         % 加载.mat文件
%         data = load(filename);
%         
%         % 初始化一个空的矩阵来保存拼接后的数据
%         featureMatrix_ADC = [];
%         
%         % 获取结构体字段数据并拼接
%         fields = fieldnames(data.TextFeature);
%         for k = 1:length(fields)
%             if ismember(fields{k}, requiredFields)
%                 structData = data.TextFeature.(fields{k});
%                 structFields = fieldnames(structData);
%                 for m = 1:length(structFields)
%                     value = structData.(structFields{m});
%                     if isnumeric(value)
%                         % 如果字段的值是数值类型，则将其转化为列向量并拼接
%                         featureMatrix_ADC = [featureMatrix_ADC; value];
%                     end
%                 end
%             end
%         end
%         
%         % 保存feature map矩阵
%         featureMaps_ADC{i} = featureMatrix_ADC;
%     end
% end
% 
% patientFolders_DCEpeak = dir(fullfile(rootDir_DCEpeak, 'MRI*'));
% 
% % 初始化结果cell数组
% featureMaps_DCEpeak = cell(length(patientFolders_DCEpeak), 1);
% 
% % 遍历每个patient文件夹
% for i = 1:length(patientFolders_DCEpeak)
%     patientDir_DCEpeak = fullfile(rootDir_DCEpeak, patientFolders_DCEpeak(i).name);
%     
%     % 获取当前patient文件夹下的所有.mat文件
%     matFiles = dir(fullfile(patientDir_DCEpeak, '*.mat'));
%     
%     for j = 1:length(matFiles)
%         % 构建文件路径
%         filename = fullfile(patientDir_DCEpeak, matFiles(j).name);
%         
%         % 加载.mat文件
%         data = load(filename);
%         
%         % 初始化一个空的矩阵来保存拼接后的数据
%         featureMatrix_DCEpeak = [];
%         
%         % 获取结构体字段数据并拼接
%         fields = fieldnames(data.TextFeature);
%         for k = 1:length(fields)
%             if ismember(fields{k}, requiredFields)
%                 structData = data.TextFeature.(fields{k});
%                 structFields = fieldnames(structData);
%                 for m = 1:length(structFields)
%                     value = structData.(structFields{m});
%                     if isnumeric(value)
%                         % 如果字段的值是数值类型，则将其转化为列向量并拼接
%                         featureMatrix_DCEpeak = [featureMatrix_DCEpeak; value];
%                     end
%                 end
%             end
%         end
%         
%         % 保存feature map矩阵
%         featureMaps_DCEpeak{i} = featureMatrix_DCEpeak;
%     end
% end
% 
% %%
% path = 'C:\Users\alexg\Desktop\study\graduationproject\code\featuremaps716\T2\MRI002\Slice_10.mat';
% test = load(path);
% 
% grayscale_matrix = test.TextFeature.grayscale;
%  
% matrix = grayscale_matrix.raw;
% 
%%
% Data_test = load('G:\NKI dataset\Feature_maps_NKI\MRDI\MRI006\Slice_12.mat');
% value_test = Data_test.('kappa');
% 
% Data_test = 'G:\NKI dataset\Data_nifti\MRI006\NIFTIs\T2.nii';
% VT = niftiread(Data_test);
% bgt = VT(:,:,12);
% bgt = rot90(bg,1);
% % value_test = Data_test.TextFeature.grayscale.raw;
% value_cropped_test = bgt(newcoordinates(2, 3):(newcoordinates(2, 4)- 1), newcoordinates(2, 1):(newcoordinates(2, 2) - 1));
% 
% test_label = table2cell(unique_labels);
% slice_label = str2num(test_label{35,2});
% 
% LA = labels_index{2};
% LA = double(LA);
% 
% label_test = slice_label(7);
% mask = LA == label_test;
% 
% 
% test_label = table2cell(unique_labels);
% slice_label = str2num(test_label{2,2});
% 
% figure;
% imagesc(mri_slice);
% colormap('gray');
% colorbar;
% axis equal;
% hold on;
% 
% for kk = 1:22
%     label_test = slice_label(kk);
%     mask = labels_resize == label_test;
%     [B, L] = bwboundaries(mask, 'noholes');
%     for k = 1:length(B)
%        boundary = B{k};
%        plot(boundary(:,2), boundary(:,1), 'w', 'LineWidth', 2);
%     end
% end
% 
% % title('Kappa with Superpixel Contours for kk from 1 to 22');
% hold off;

% [B, L] = bwboundaries(mask, 'noholes');
% for k = 1:length(B)
%    boundary = B{k};
%    plot(boundary(:,2), boundary(:,1), 'w', 'LineWidth', 2)
% end
% title('Kappa with Superpixel Contours');
% hold off;

%%