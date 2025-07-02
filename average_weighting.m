% % 假设我们有以下变量：
% % superpixels_features - 一个包含所有superpixel特征的cell数组，每个cell是一个patient的特征矩阵 (NxM, N是superpixel数量，M是特征数量)
% % tumor_proportion - 一个包含所有superpixel的肿瘤区域占比的cell数组，每个cell是一个patient的占比向量 (Nx1)
% % labels - 每个patient的标签 (1x35)
% 
% % 读取数据示例（这里假设数据已经加载到工作区）
% % load('superpixels_features.mat'); % 加载superpixel特征
% % load('tumor_proportion.mat'); % 加载肿瘤区域占比
% % load('labels.mat'); % 加载标签
% 
% num_patients = length(superpixels_features);
% patient_features = cell(num_patients, 1);
% 
% for i = 1:num_patients
%     features = superpixels_features{i};
%     proportions = tumor_proportion{i};
%     
%     % 计算加权平均特征
%     weighted_features = bsxfun(@times, features, proportions);
%     average_features = sum(weighted_features, 1) / sum(proportions);
%     
%     % 存储结果
%     patient_features{i} = average_features;
% end
% 
% % 将合并后的特征转换为矩阵
% X = cell2mat(patient_features);
% y = labels;
% 
% % 训练分类器（使用例如SVM）
% SVMModel = fitcsvm(X, y);
% 
% % 进行交叉验证
% CVSVMModel = crossval(SVMModel);
% classLoss = kfoldLoss(CVSVMModel);
% 
% fprintf('Cross-validated classification loss: %f\n', classLoss);






% 假设我们有以下变量：
% superpixels_features - 一个包含所有superpixel特征的cell数组，每个cell是一个patient的特征矩阵 (NxM, N是superpixel数量，M是特征数量)
% tumor_proportion - 一个包含所有superpixel的肿瘤区域占比的cell数组，每个cell是一个patient的占比向量 (Nx1)
% labels - 每个patient的标签 (1x35)

% 读取数据
file_path = 'C:\Users\alexg\Desktop\study\graduation project\code\unique_labels_76.xlsx';
Data = readtable(file_path);
proportions = Data.Tumor_Proportions(:,3);

superpixels_features = load('tumor_proportions.mat'); % 加载superpixel特征和肿瘤区域占比

num_patients = height(Data);
patient_features = cell(num_patients, 1);

for i = 1:num_patients
    proportions = proportions{i};
    proportions = proportions(2:end-1);
    proportions_array = str2num(proportions);
    
    % 计算加权平均特征
    weighted_features = bsxfun(@times, features, proportions);
    average_features = sum(weighted_features, 1) / sum(proportions);
    
    % 存储结果
    tumor_proportions{i} = proportions_array;
end

base_path = 'C:\Users\alexg\Desktop\study\graduation project\code\feature maps\ADC\';

for i = 1:num_patients
    patient = T{i,'PatientName'};
    whole_path = fullfile(base_path, patient);
    mat_files = dir(fullfile(whole_path, '*.mat'));
    mat_file_path = fullfile(whole_path, mat_files(1).name);
    features = load(mat_file_path);

    features = features.TextFeature;
    
end



% 将合并后的特征转换为矩阵
X = cell2mat(patient_features);
y = labels;

% 训练分类器（使用例如SVM）
SVMModel = fitcsvm(X, y);

% 进行交叉验证
CVSVMModel = crossval(SVMModel);
classLoss = kfoldLoss(CVSVMModel);

fprintf('Cross-validated classification loss: %f\n', classLoss);
