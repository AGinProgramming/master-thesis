% % 假设文件名为 'data.mat'
% filename = 'C:\Users\alexg\Desktop\study\graduation project\code\feature maps\ADC\MRI002\Slice_10.mat';
% 
% % 加载 .mat 文件
% data = load(filename);
% 
% % 假设 .mat 文件中包含一个名为 'myStruct' 的结构体
% % 访问结构体中的字段
% Global = data.TextFeature.Global;
% GLCM = data.TextFeature.GLCM;
% NGTDM = data.TextFeature.NGTDM;
% GLSZM = data.TextFeature.GLSZM;
% GLRLM = data.TextFeature.GLRLM;
% 
% 
% 
% % 如果需要查看结构体的所有字段，可以使用 'fieldnames' 函数
% fields = fieldnames(data.TextFeature);
% disp(fields);
% 
% % 迭代访问结构体的所有字段
% for i = 1:length(fields)
%     fieldName = fields{i};
%     fieldValue = data.TextFeature.(fieldName);
%     disp(['Field Name: ', fieldName]);
%     disp('Field Value: ');
%     disp(fieldValue);
% end


%% first try
% % 假设根目录
% rootDir = 'C:\Users\alexg\Desktop\study\graduation project\code\feature maps\ADC\';
% 
% % 获取根目录下所有子文件夹
% patientFolders = dir(fullfile(rootDir, 'MRI*'));
% 
% % 初始化结果cell数组
% featureMaps = cell(length(patientFolders), 1);
% 
% % 遍历每个patient文件夹
% for i = 1:length(patientFolders)
%     patientDir = fullfile(rootDir, patientFolders(i).name);
%     
%     % 获取当前patient文件夹下的所有.mat文件
%     matFiles = dir(fullfile(patientDir, '*.mat'));
%     
%     for j = 1:length(matFiles)
%         % 构建文件路径
%         filename = fullfile(patientDir, matFiles(j).name);
%         
%         % 加载.mat文件
%         data = load(filename);
%         
%         % 获取结构体字段数据
%         Global = data.TextFeature.Global;
%         GLCM = data.TextFeature.GLCM;
%         NGTDM = data.TextFeature.NGTDM;
%         GLSZM = data.TextFeature.GLSZM;
%         GLRLM = data.TextFeature.GLRLM;
%         
%         % 将小结构体中的数据拼接为矩阵（假设每个结构体包含的数据可以通过字段访问）
%         featureMatrix = [Global(:); GLCM(:); NGTDM(:); GLSZM(:); GLRLM(:)];
%         
%         % 保存feature map矩阵
%         featureMaps{i} = featureMatrix;
%     end
% end

% featureMaps cell数组中包含每个patient文件夹的feature map矩阵

%% second try
% % 假设根目录
% rootDir = 'C:\Users\alexg\Desktop\study\graduation project\code\feature maps\';
% subFolders = {'ADC', 'DCEpeak', 'DCEwashin', 'T2'};
% 
% % 获取根目录下所有子文件夹
% patientFolders = dir(fullfile(rootDir, subFolders{1}, 'MRI*'));
% 
% % 初始化结果cell数组
% featureMaps = cell(length(patientFolders), length(subFolders));
% requiredFields = {'Global', 'GLCM', 'NGTDM', 'GLSZM', 'GLRLM'};
% 
% % 遍历每个子文件夹
% for sf = 1:length(subFolders)
%     subFolderPath = fullfile(rootDir, subFolders{sf});
%     
%     % 遍历每个patient文件夹
%     for i = 1:length(patientFolders)
%         patientDir = fullfile(subFolderPath, patientFolders(i).name);
%         
%         % 获取当前patient文件夹下的所有.mat文件
%         matFiles = dir(fullfile(patientDir, '*.mat'));
%         
%         for j = 1:length(matFiles)
%             % 构建文件路径
%             filename = fullfile(patientDir, matFiles(j).name);
%             
%             % 加载.mat文件
%             data = load(filename);
%             
%             % 初始化一个空的矩阵来保存拼接后的数据
%             featureMatrix = [];
%             
%             % 获取结构体字段数据并拼接
%             fields = fieldnames(data.TextFeature);
%             for k = 1:length(fields)
%                 if ismember(fields{k}, requiredFields)
%                     structData = data.TextFeature.(fields{k});
%                     structFields = fieldnames(structData);
%                     for m = 1:length(structFields)
%                         value = structData.(structFields{m});
%                         if isnumeric(value)
%                             % 如果字段的值是数值类型，则将其转化为列向量并拼接
%                             featureMatrix = [featureMatrix; value];
%                         end
%                     end
%                 end
%             end
%             
%             % 保存feature map矩阵
%             featureMaps{i, sf} = featureMatrix;
%         end
%     end
% end
% 
% % 读取数据
% file_path = 'C:\Users\alexg\Desktop\study\graduation project\code\unique_labels_76.xlsx';
% ID_path = 'G:\Annekoos Schaap - code\Data\All_imaging_features_NKI.xlsx';
% ID = readtable(ID_path);
% Data = readtable(file_path);
% proportions_data = Data.TumorProportions;
% 
% num_patients = height(Data);
% patient_features = cell(num_patients, 1);
% headers = {};
% 
% % 遍历每个patient
% for i = 1:num_patients
%     % 解析proportions
%     proportions = proportions_data{i};
%     proportions = proportions(2:end-1);
%     proportions_array = str2num(proportions);
%     
%     % 初始化一个用于存储所有文件夹特征的数组
%     all_features = [];
%     
%     % 遍历每个子文件夹
%     for sf = 1:length(subFolders)
%         % 获取当前patient的特征数据
%         features = featureMaps{i, sf};
%         
%         % 初始化一个43行的矩阵来保存加权平均特征
%         average_features = zeros(43, 1);
%         
%         % 对每一个texture feature计算加权平均
%         for j = 1:43
%             texture_feature = features(j, :);
%             if isnumeric(texture_feature) && size(texture_feature, 2) == length(proportions_array)
%                 % 计算加权平均特征
%                 weighted_features = bsxfun(@times, texture_feature, proportions_array);
%                 average_features(j) = sum(weighted_features) / sum(proportions_array);
%             end
%         end
%         
%         % 存储结果
%         all_features = [all_features; average_features];
%         
%         % 更新header
%         if i == 1
%             for field_idx = 1:length(requiredFields)
%                 headers = [headers; strcat(subFolders{sf}, '_', requiredFields{field_idx})];
%             end
%         end
%     end
%     
%     patient_features{i} = all_features;
% end
% 
% % 将合并后的特征转换为矩阵
% X = cell2mat(patient_features');
% 
% % 创建表格
% featureTable = array2table(X, 'VariableNames', headers);
% 
% % 添加Patient ID列
% featureTable.PatientID = ID.Patient_name;
% 
% % 重新排列列顺序
% featureTable = featureTable(:, [{'PatientID'}, headers']);
% 
% %% data save
% % 将矩阵保存为.mat文件
% output_file = 'G:\Code\Hesheng Gong - code\feature_map_all.mat';
% save(output_file, 'featureTable');
% 
% fprintf('Feature matrix saved to %s\n', output_file);
% 
% output_file = 'G:\Code\Hesheng Gong - code\feature_map_all.xlsx';
% writetable(featureTable, output_file);
% 
% fprintf('Feature matrix saved to %s\n', output_file);




%%
% 假设根目录
% rootDir = 'C:\Users\alexg\Desktop\study\graduation project\code\feature maps\';
% subFolders = {'ADC', 'DCEpeak', 'DCEwashin', 'T2'};

rootDir_ADC = 'C:\Users\alexg\Desktop\study\graduationproject\code\featuremaps729\ADC\';
rootDir_DCEpeak = 'C:\Users\alexg\Desktop\study\graduationproject\code\featuremaps729\DCEpeak\';
rootDir_DCEwashin = 'C:\Users\alexg\Desktop\study\graduationproject\code\featuremaps729\DCEwashin\';
rootDir_T2 = 'C:\Users\alexg\Desktop\study\graduationproject\code\featuremaps729\T2\';


% 获取根目录下所有子文件夹
% patientFolders = dir(fullfile(rootDir, subFolders{1}, 'MRI*'));
patientFolders_ADC = dir(fullfile(rootDir_ADC, 'MRI*'));

% 初始化结果cell数组
featureMaps_ADC = cell(length(patientFolders_ADC), 1);

requiredFields = {'Global', 'GLCM', 'NGTDM', 'GLSZM', 'GLRLM'};

% 遍历每个patient文件夹
for i = 1:length(patientFolders_ADC)
    patientDir_ADC = fullfile(rootDir_ADC, patientFolders_ADC(i).name);
    
    % 获取当前patient文件夹下的所有.mat文件
    matFiles = dir(fullfile(patientDir_ADC, '*.mat'));
    
    for j = 1:length(matFiles)
        % 构建文件路径
        filename = fullfile(patientDir_ADC, matFiles(j).name);
        
        % 加载.mat文件
        data = load(filename);
        
        % 初始化一个空的矩阵来保存拼接后的数据
        featureMatrix_ADC = [];
        
        % 获取结构体字段数据并拼接
        fields = fieldnames(data.TextFeature);
        for k = 1:length(fields)
            if ismember(fields{k}, requiredFields)
                structData = data.TextFeature.(fields{k});
                structFields = fieldnames(structData);
                for m = 1:length(structFields)
                    value = structData.(structFields{m});
                    if isnumeric(value)
                        % 如果字段的值是数值类型，则将其转化为列向量并拼接
                        featureMatrix_ADC = [featureMatrix_ADC; value];
                    end
                end
            end
        end
        
        % 保存feature map矩阵
        featureMaps_ADC{i} = featureMatrix_ADC;
    end
end
%%
patientFolders_DCEpeak = dir(fullfile(rootDir_DCEpeak, 'MRI*'));

% 初始化结果cell数组
featureMaps_DCEpeak = cell(length(patientFolders_DCEpeak), 1);

% 遍历每个patient文件夹
for i = 1:length(patientFolders_DCEpeak)
    patientDir_DCEpeak = fullfile(rootDir_DCEpeak, patientFolders_DCEpeak(i).name);
    
    % 获取当前patient文件夹下的所有.mat文件
    matFiles = dir(fullfile(patientDir_DCEpeak, '*.mat'));
    
    for j = 1:length(matFiles)
        % 构建文件路径
        filename = fullfile(patientDir_DCEpeak, matFiles(j).name);
        
        % 加载.mat文件
        data = load(filename);
        
        % 初始化一个空的矩阵来保存拼接后的数据
        featureMatrix_DCEpeak = [];
        
        % 获取结构体字段数据并拼接
        fields = fieldnames(data.TextFeature);
        for k = 1:length(fields)
            if ismember(fields{k}, requiredFields)
                structData = data.TextFeature.(fields{k});
                structFields = fieldnames(structData);
                for m = 1:length(structFields)
                    value = structData.(structFields{m});
                    if isnumeric(value)
                        % 如果字段的值是数值类型，则将其转化为列向量并拼接
                        featureMatrix_DCEpeak = [featureMatrix_DCEpeak; value];
                    end
                end
            end
        end
        
        % 保存feature map矩阵
        featureMaps_DCEpeak{i} = featureMatrix_DCEpeak;
    end
end
%%
patientFolders_DCEwashin = dir(fullfile(rootDir_DCEwashin, 'MRI*'));

% 初始化结果cell数组
featureMaps_DCEwashin = cell(length(patientFolders_DCEwashin), 1);

% 遍历每个patient文件夹
for i = 1:length(patientFolders_DCEwashin)
    patientDir_DCEwashin = fullfile(rootDir_DCEwashin, patientFolders_DCEwashin(i).name);
    
    % 获取当前patient文件夹下的所有.mat文件
    matFiles = dir(fullfile(patientDir_DCEwashin, '*.mat'));
    
    for j = 1:length(matFiles)
        % 构建文件路径
        filename = fullfile(patientDir_DCEwashin, matFiles(j).name);
        
        % 加载.mat文件
        data = load(filename);
        
        % 初始化一个空的矩阵来保存拼接后的数据
        featureMatrix_DCEwashin = [];
        
        % 获取结构体字段数据并拼接
        fields = fieldnames(data.TextFeature);
        for k = 1:length(fields)
            if ismember(fields{k}, requiredFields)
                structData = data.TextFeature.(fields{k});
                structFields = fieldnames(structData);
                for m = 1:length(structFields)
                    value = structData.(structFields{m});
                    if isnumeric(value)
                        % 如果字段的值是数值类型，则将其转化为列向量并拼接
                        featureMatrix_DCEwashin = [featureMatrix_DCEwashin; value];
                    end
                end
            end
        end
        
        % 保存feature map矩阵
        featureMaps_DCEwashin{i} = featureMatrix_DCEwashin;
    end
end
%%
patientFolders_T2 = dir(fullfile(rootDir_T2, 'MRI*'));

% 初始化结果cell数组
featureMaps_T2 = cell(length(patientFolders_T2), 1);

% 遍历每个patient文件夹
for i = 1:length(patientFolders_T2)
    patientDir_T2 = fullfile(rootDir_T2, patientFolders_T2(i).name);
    
    % 获取当前patient文件夹下的所有.mat文件
    matFiles = dir(fullfile(patientDir_T2, '*.mat'));
    
    for j = 1:length(matFiles)
        % 构建文件路径
        filename = fullfile(patientDir_T2, matFiles(j).name);
        
        % 加载.mat文件
        data = load(filename);
        
        % 初始化一个空的矩阵来保存拼接后的数据
        featureMatrix_T2 = [];
        
        % 获取结构体字段数据并拼接
        fields = fieldnames(data.TextFeature);
        for k = 1:length(fields)
            if ismember(fields{k}, requiredFields)
                structData = data.TextFeature.(fields{k});
                structFields = fieldnames(structData);
                for m = 1:length(structFields)
                    value = structData.(structFields{m});
                    if isnumeric(value)
                        % 如果字段的值是数值类型，则将其转化为列向量并拼接
                        featureMatrix_T2 = [featureMatrix_T2; value];
                    end
                end
            end
        end
        
        % 保存feature map矩阵
        featureMaps_T2{i} = featureMatrix_T2;
    end
end
% featureMaps cell数组中包含每个patient文件夹的feature map矩阵
%%
% 读取数据
file_path = 'C:\Users\alexg\Desktop\study\graduation project\code\unique_labels_729.xlsx';
Data = readtable(file_path);
proportions_data = Data.TumorProportions;

num_patients = height(Data);
patient_features = cell(num_patients, 1);

% 遍历每个patient
for i = 1:num_patients
    % 解析proportions
    proportions = proportions_data{i};
    proportions = proportions(2:end-1);
    disp(numel(proportions));
    proportions_array = str2num(proportions);
    
    % 获取当前patient的特征数据
    features = featureMaps_ADC{i};
    
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
X_ADC = cell2mat(patient_features');
%%
% 遍历每个patient
for i = 1:num_patients
    % 解析proportions
    proportions = proportions_data{i};
    proportions = proportions(2:end-1);
    disp(numel(proportions));
    proportions_array = str2num(proportions);
    
    % 获取当前patient的特征数据
    features = featureMaps_DCEpeak{i};
    
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
X_DCEpeak = cell2mat(patient_features');
%%
% 遍历每个patient
for i = 1:num_patients
    % 解析proportions
    proportions = proportions_data{i};
    proportions = proportions(2:end-1);
    disp(numel(proportions));
    proportions_array = str2num(proportions);
    
    % 获取当前patient的特征数据
    features = featureMaps_T2{i};
    
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
X_T2 = cell2mat(patient_features');
%%
% 遍历每个patient
for i = 1:num_patients
    % 解析proportions
    proportions = proportions_data{i};
    proportions = proportions(2:end-1);
    disp(numel(proportions));
    proportions_array = str2num(proportions);
    
    % 获取当前patient的特征数据
    features = featureMaps_DCEwashin{i};
    
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
X_DCEwashin = cell2mat(patient_features');
%% data save
% 将矩阵保存为.mat文件
output_file_DCEwashin = 'D:\code备份\7.29\Hesheng Gong - code\feature_map_all_DCEwashin729.mat';
save(output_file_DCEwashin, 'X_DCEwashin');

fprintf('Feature matrix saved to %s\n', output_file_DCEwashin);

output_file_DCEwashin = 'D:\code备份\7.29\Hesheng Gong - code\feature_map_all_DCEwashin729.xlsx';
writematrix(X_DCEwashin, output_file_DCEwashin);

fprintf('Feature matrix saved to %s\n', output_file_DCEwashin);
%%
output_file_DCEpeak = 'D:\code备份\7.29\Hesheng Gong - code\feature_map_all_DCEpeak729.mat';
save(output_file_DCEpeak, 'X_DCEpeak');

fprintf('Feature matrix saved to %s\n', output_file_DCEpeak);

output_file_DCEpeak = 'D:\code备份\7.29\Hesheng Gong - code\feature_map_all_DCEpeak729.xlsx';
writematrix(X_DCEpeak, output_file_DCEpeak);

fprintf('Feature matrix saved to %s\n', output_file_DCEpeak);
%%
output_file_ADC = 'D:\code备份\7.29\Hesheng Gong - code\feature_map_all_ADC729.mat';
save(output_file_ADC, 'X_ADC');

fprintf('Feature matrix saved to %s\n', output_file_ADC);

output_file_ADC = 'D:\code备份\7.29\Hesheng Gong - code\feature_map_all_ADC729.xlsx';
writematrix(X_ADC, output_file_ADC);

fprintf('Feature matrix saved to %s\n', output_file_ADC);
%%
output_file_T2 = 'D:\code备份\7.29\Hesheng Gong - code\feature_map_all_T2729.mat';
save(output_file_T2, 'X_T2');

fprintf('Feature matrix saved to %s\n', output_file_T2);

output_file_T2 = 'D:\code备份\7.29\Hesheng Gong - code\feature_map_all_T2729.xlsx';
writematrix(X_T2, output_file_T2);

fprintf('Feature matrix saved to %s\n', output_file_T2);
%%
rootDir_ADC = 'C:\Users\alexg\Desktop\study\graduationproject\code\featuremaps729\ADC\';
rootDir_DCEpeak = 'C:\Users\alexg\Desktop\study\graduationproject\code\featuremaps729\DCEpeak\';
rootDir_DCEwashin = 'C:\Users\alexg\Desktop\study\graduationproject\code\featuremaps729\DCEwashin\';
rootDir_T2 = 'C:\Users\alexg\Desktop\study\graduationproject\code\featuremaps729\T2\';

requiredFields = {'tumor_value'};

patientFolders_ADC = dir(fullfile(rootDir_ADC, 'MRI*'));

featureMaps_ADC = cell(length(patientFolders_ADC), 1);

featureMatrix_ADC = [];

% 遍历每个patient文件夹
for i = 1:length(patientFolders_ADC)
    patientDir_ADC = fullfile(rootDir_ADC, patientFolders_ADC(i).name);
    
    % 获取当前patient文件夹下的所有.mat文件
    matFiles = dir(fullfile(patientDir_ADC, '*.mat'));
    
    for j = 1:length(matFiles)
        % 构建文件路径
        filename = fullfile(patientDir_ADC, matFiles(j).name);
        
        % 加载.mat文件
        data = load(filename);
        
        % 获取结构体字段数据并拼接
        grayscale_weighted = data.TextFeature.tumor_value;

        featureMatrix_ADC = [featureMatrix_ADC; grayscale_weighted];
    end
end

patientFolders_DCEpeak = dir(fullfile(rootDir_DCEpeak, 'MRI*'));

featureMaps_DCEpeak = cell(length(patientFolders_DCEpeak), 1);

featureMatrix_DCEpeak = [];

% 遍历每个patient文件夹
for i = 1:length(patientFolders_DCEpeak)
    patientDir_DCEpeak = fullfile(rootDir_DCEpeak, patientFolders_DCEpeak(i).name);
    
    % 获取当前patient文件夹下的所有.mat文件
    matFiles = dir(fullfile(patientDir_DCEpeak, '*.mat'));
    
    for j = 1:length(matFiles)
        % 构建文件路径
        filename = fullfile(patientDir_DCEpeak, matFiles(j).name);
        
        % 加载.mat文件
        data = load(filename);

        % 获取结构体字段数据并拼接
        grayscale_weighted = data.TextFeature.tumor_value;

        featureMatrix_DCEpeak = [featureMatrix_DCEpeak; grayscale_weighted];
    end
end

patientFolders_DCEwashin = dir(fullfile(rootDir_DCEwashin, 'MRI*'));

featureMaps_DCEwashin = cell(length(patientFolders_DCEwashin), 1);

featureMatrix_DCEwashin = [];

% 遍历每个patient文件夹
for i = 1:length(patientFolders_DCEwashin)
    patientDir_DCEwashin = fullfile(rootDir_DCEwashin, patientFolders_DCEwashin(i).name);
    
    % 获取当前patient文件夹下的所有.mat文件
    matFiles = dir(fullfile(patientDir_DCEwashin, '*.mat'));
    
    for j = 1:length(matFiles)
        % 构建文件路径
        filename = fullfile(patientDir_DCEwashin, matFiles(j).name);
        
        % 加载.mat文件
        data = load(filename);
        
        % 获取结构体字段数据并拼接
        grayscale_weighted = data.TextFeature.tumor_value;

        featureMatrix_DCEwashin = [featureMatrix_DCEwashin; grayscale_weighted];
    end
end

patientFolders_T2 = dir(fullfile(rootDir_T2, 'MRI*'));

featureMaps_T2 = cell(length(patientFolders_T2), 1);

featureMatrix_T2 = [];

% 遍历每个patient文件夹
for i = 1:length(patientFolders_T2)
    patientDir_T2 = fullfile(rootDir_T2, patientFolders_T2(i).name);
    
    % 获取当前patient文件夹下的所有.mat文件
    matFiles = dir(fullfile(patientDir_T2, '*.mat'));
    
    for j = 1:length(matFiles)
        % 构建文件路径
        filename = fullfile(patientDir_T2, matFiles(j).name);
        
        % 加载.mat文件
        data = load(filename);
        
        % 获取结构体字段数据并拼接
        grayscale_weighted = data.TextFeature.tumor_value;
    zAAAAAAAA``` VCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
        featureMatrix_T2 = [featureMatrix_T2; grayscale_weighted];
    end
end

gray_scale_map = [featureMatrix_ADC, featureMatrix_DCEpeak, featureMatrix_DCEwashin, featureMatrix_T2];
save_dir = 'D:\code备份\7.29\Hesheng Gong - code';
outputFileName = 'grayscale_map729.xlsx';
fullFileName = fullfile(save_dir, outputFileName);
xlswrite(outputFileName, gray_scale_map);
disp(['Feature matrix saved to ', outputFileName]);












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
% 
% %%
% % 初始化结果cell数组
% featureMaps_ADC = cell(length(targetFolders), 1);
% 
% requiredFields = {'A', 'kappa', 'kep', 'mu', 't0_est'};
% 
% % 遍历每个patient文件夹
% for i = 1:length(targetFolders)
%     folder = targetFolders(i);
%     patientDir_MRDI = folder{1};
%     
%     % 获取当前patient文件夹下的所有.mat文件
%     matFiles = dir(fullfile(patientDir_MRDI, '*.mat'));
%     
%     for j = 1:length(matFiles)
%         % 构建文件路径
%         filename = fullfile(patientDir_MRDI, matFiles(j).name);
%         
%         % 加载.mat文件
%         data = load(filename);
%         
%         % 初始化一个空的矩阵来保存拼接后的数据
%         featureMatrix_MRDI = [];
%         
%         % 获取结构体字段数据并拼接
%         fields = fieldnames(data);
%         for k = 1:length(fields)
%             if ismember(fields{k}, requiredFields)
%                 structData = data.(fields{k});
%                 structFields = fieldnames(structData);
%                 for m = 1:length(structFields)
%                     value = structData.(structFields{m});
%                     if isnumeric(value)
%                         % 如果字段的值是数值类型，则将其转化为列向量并拼接
%                         featureMatrix_MRDI = [featureMatrix_MRDI; value];
%                     end
%                 end
%             end
%         end
%         
%         % 保存feature map矩阵
%         featureMaps_MRDI{i} = featureMatrix_MRDI;
%     end
% end