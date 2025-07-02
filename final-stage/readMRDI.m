% coordinate = load('all_patients_data.mat', 'all_patients_data');
% coordinates = coordinate.all_patients_data;
% 
% newcoordinates = [];
% 
% for i = 1:height(coordinates)
%     minX = coordinates(i, 1) - 20;
%     minY = coordinates(i, 2) - 25;
%     maxX = coordinates(i, 3) + 20;
%     maxY = coordinates(i, 4) + 10;
%     newcoordinates = [newcoordinates; minX, maxX, minY, maxY];
%     %(minY:maxY, minX:maxX)
% end




% 
% %% MRDI operations
% rootDir_MRDI = 'G:\NKI dataset\Feature_maps_NKI\MRDI';
% patientFolders_MRDI = dir(fullfile(rootDir_MRDI, 'MRI*'));
% 
% resultsFile = 'C:\Users\alexg\Desktop\study\graduation project\code\results5_6.xlsx';
% patientData = readtable(resultsFile);
% targetPatientIDs = patientData.PatientID;
% 
% targetFolders = {};
% for i = 1:length(targetPatientIDs)
%     for j = 1:length(patientFolders_MRDI)
%         folderName = patientFolders_MRDI(j).name;
%         if contains(folderName, targetPatientIDs{i})
%             targetFolders{end+1} = fullfile(rootDir_MRDI, folderName);
%             break; % 找到对应文件夹后跳出内层循环
%         end
%     end
% end

%%
% 初始化结果cell数组
featureMaps_MRDI = cell(length(targetFolders), 1);

requiredFields = {'A', 'kappa', 'kep', 'mu', 't0_est'};

% 遍历每个patient文件夹
for i = 1:length(targetFolders)
    folder = targetFolders(i);
    patientDir_MRDI = folder{1};
    
    % 获取当前patient文件夹下的所有.mat文件
    matFiles = dir(fullfile(patientDir_MRDI, '*.mat'));
    
    for j = 1:length(matFiles)
        % 构建文件路径
        filename = fullfile(patientDir_MRDI, matFiles(j).name);
        
        % 加载.mat文件
        data = load(filename);
        
        % 初始化一个空的矩阵来保存拼接后的数据
        featureMatrix_MRDI = [];

         % 遍历 requiredFields
        for k = 1:length(requiredFields)
            fieldName = requiredFields{k};
            if isfield(data, fieldName)
                value = data.(fieldName);
                value_cropped = value(newcoordinates(i, 3):(newcoordinates(i, 4)- 1), newcoordinates(i, 1):(newcoordinates(i, 2) - 1));
                featureMatrix_MRDI = [featureMatrix_MRDI; value_cropped(:)];
                %有点懵了，要做：70行不要，改为在这个cropped的value图中按照superpixel计算median值，并按照average_weight计算整张图的值
            end
        end

         % 保存feature map矩阵
        featureMaps_MRDI{i} = [featureMaps_MRDI{i}, featureMatrix_MRDI]; 
        
%         % 获取结构体字段数据并拼接
%         fields = fieldnames(data);
%         for k = 1:length(fields)
%             if ismember(fields{k}, requiredFields)
%                 structData = data.(fields{k});
%                 structFields = fieldnames(structData);
%                 for m = 1:length(structFields)
%                     value = structData.(structFields{m});
%                     value_cropped = value(newcoordinates(i, 3):(newcoordinates(i, 4)- 1), newcoordinates(i, 1):(newcoordinates(i, 2) - 1));
%                     if isnumeric(value)
%                         % 如果字段的值是数值类型，则将其转化为列向量并拼接
%                         featureMatrix_MRDI = [featureMatrix_MRDI; value_cropped];
%                     end
%                 end
%             end
%         end
        
%         % 保存feature map矩阵
%         featureMaps_MRDI{i} = featureMatrix_MRDI;
    end
end

%%
file_path = 'C:\Users\alexg\Desktop\study\graduation project\code\unique_labels_76.xlsx';
Data = readtable(file_path);
unique_labels_cell = table2cell(Data);
proportions_data = Data.TumorProportions;

num_patients = height(Data);
patient_features = cell(num_patients, 1);

% 遍历每个patient
for i = 1:num_patients
    % 解析proportions
    unique_labels_in_slice = str2num(unique_labels_cell{i,2});
    weights_in_slice = str2num(unique_labels_cell{i,3});
    median_values = zeros(size(unique_labels_in_slice));
    weights = zeros(size(unique_labels_in_slice));
    for l = 1:length(unique_labels_in_slice)
        label = unique_labels_in_slice(l);
        mask = labels_resize == label;
        median_values(l) = median(mri_slice(mask));
        weights(l) = weights_in_slice(l);
    end

    proportions = proportions_data{i};
    proportions = proportions(2:end-1);
    disp(numel(proportions));
    proportions_array = str2num(proportions);
    
    % 获取当前patient的特征数据
    features = featureMaps_MRDI{i};
    
    % 初始化一个43行的矩阵来保存加权平均特征
    average_features = zeros(43, 1);
    
    % 对每一个texture feature计算加权平均
    for j = 1:43
        texture_feature = features(j, :);
        if isnumeric(texture_feature) && size(texture_feature, 2) == length(proportions_array)
            % 计算加权平均特征
            weighted_features = bsxfun(@times, texture_feature, proportions_array);
            average_features(j) = sum(weighted_features) / sum(proportions_array);
        end
    end
    
    % 存储结果
    patient_features{i} = average_features;
end

% 将合并后的特征转换为矩阵
X_MRDI = cell2mat(patient_features');

%% data save
% 将矩阵保存为.mat文件
output_file_MRDI = 'G:\Code\Hesheng Gong - code\feature_map_all_MRDI.mat';
save(output_file_MRDI, 'X_MRDI');

fprintf('Feature matrix saved to %s\n', output_file_MRDI);

output_file_MRDI = 'G:\Code\Hesheng Gong - code\feature_map_all_MRDI.xlsx';
writematrix(X_MRDI, output_file_MRDI);

fprintf('Feature matrix saved to %s\n', output_file_MRDI);