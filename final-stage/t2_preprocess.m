function [mri_gaussian_normalized, deli_resized] = t2_preprocess(mri_path,deli_path, CNN_slice)
   mri = double(niftiread(mri_path));
   mri = flip(rot90(mri,3),2);
%    deli = double(niftiread(deli_path))>0;
%    deli = 0;
   deli_data = load(deli_path);
   fieldnames_deli = fieldnames(deli_data);
   deli = deli_data.(fieldnames_deli{1});
%    deli = flip(rot90(deli,3),2);
   deli = double(deli > 0);
%    disp(size(deli))
%    disp(size(mri))

   deli_resized = zeros(size(mri));
   mri_gaussian = zeros(size(mri));
%    disp(size(deli_resized))
%    disp(size(mri_gaussian))

   deli_resized = imresize(deli, [size(mri, 1) size(mri, 2)], "nearest");
   mri_gaussian(:,:,CNN_slice) = imgaussfilt(mri(:,:,CNN_slice), 0.5, 'FilterSize', 3);

%    for kk = 1:size(deli,3)
% %         mri2(:,:,kk) = rot90(mri(:,:,kk),3);
%         disp(kk)
%         deli_resized(:,:,kk) = imresize(deli(:,:,kk), [size(mri,1) size(mri,2)] ,"nearest");
%         mri_gaussian(:,:,kk) = imgaussfilt(mri(:,:,kk), 0.5, 'FilterSize',3);
%    end
%    imshow(deli_resized(:,:,1)); % 显示调整大小后的掩码
%    imshow(mri(:,:,CNN_slice)); % 显示对应的MRI切片

   % 平移高斯滤波后的值，使其最小值为0
   mri_gaussian_shifted = mri_gaussian(:,:,CNN_slice) - min(mri_gaussian(:,:,CNN_slice), [], 'all');
    
   % 重新计算平均值和标准差
   mri_filtered_shifted = mri_gaussian_shifted .* deli_resized;
   mri_filtered_shifted(mri_filtered_shifted == 0) = nan;
   mri_filtered_shifted_vec = mri_filtered_shifted(:);
   avg_shifted = mean(mri_filtered_shifted_vec, 'omitnan');
   sdev_shifted = std(mri_filtered_shifted_vec, 'omitnan');


%    mri_filtered = mri(:,:,CNN_slice).*deli_resized;
%    mri_filtered(mri_filtered == 0) = nan;
%    mri_filtered_vec = mri_filtered(:);
%    avg = mean(mri_filtered_vec, 'omitnan');
%    sdev = std(mri_filtered_vec, 'omitnan');

   disp(['Average: ', num2str(avg_shifted)]);
   disp(['Standard Deviation: ', num2str(sdev_shifted)]);

%    % 归一化处理
%    mri_gaussian_normalized_shifted = (mri_gaussian_shifted - avg_shifted) / sdev_shifted;
%    mri_gaussian_normalized = mri_gaussian_normalized_shifted;

   % 直接调整归一化公式，确保所有值非负
   mri_gaussian_normalized(:,:,CNN_slice) = (mri_gaussian(:,:,CNN_slice) - avg_shifted + abs(min(mri_gaussian(:,:,CNN_slice), [], 'all'))) / sdev_shifted;
   mri_gaussian_normalized = mri_gaussian_normalized(:,:,CNN_slice);


%    mri_gaussian_normalized = (mri_gaussian-avg)/sdev;

%    mri_gaussian_normalized = zeros(size(mri));
%    mri_gaussian_normalized(:,:,CNN_slice) = (mri_gaussian(:,:,CNN_slice) - avg) / sdev;
%    mri_gaussian_normalized = mri_gaussian_normalized(:,:,CNN_slice);

   min_val = min(mri_gaussian(:,:,CNN_slice), [], 'all');
   max_val = max(mri_gaussian(:,:,CNN_slice), [], 'all');
   disp(['Min value after Gaussian filter: ', num2str(min_val)]);
   disp(['Max value after Gaussian filter: ', num2str(max_val)]);

%    disp(mri_filtered)
%    disp(sdev)

end