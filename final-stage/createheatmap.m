% 加载保存的 .mat 文件
feature_map_all = load('feature_map_all.mat');  

fields = fieldnames(feature_map_all);
disp('Structure fields:');
disp(fields);

data = feature_map_all.X; 

if isnumeric(data) && ismatrix(data)
    heatmap(data);

    title('Heatmap of feature_map_all');
else
    error('Extracted data must be a 2-dimensional numeric matrix.');
end
