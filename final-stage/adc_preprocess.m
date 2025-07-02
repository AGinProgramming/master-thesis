function [mri_gaussian, deli_resized] = adc_preprocess(mri_path,deli_path,CNN_slice)
   mri = double(niftiread(mri_path));
   mri = flip(rot90(mri,3),2);
%    deli = double(niftiread(deli_path))>0;
   deli_data = load(deli_path);
   fieldnames_deli = fieldnames(deli_data);
   deli = deli_data.(fieldnames_deli{1});
%    deli = flip(rot90(deli,3),2);
   deli = double(deli > 0);

   deli_resized = zeros(size(mri));
   mri_gaussian = zeros(size(mri));

   deli_resized = imresize(deli, [size(mri, 1) size(mri, 2)], "nearest");
   mri_gaussian = imgaussfilt(mri, 0.5, 'FilterSize', 3);

   mri_gaussian = mri_gaussian(:,:,CNN_slice);


%    for kk = 1:size(deli,3)
% %         mri(:,:,kk) = rot90(mri(:,:,kk),3);
%         deli_resized(:,:,kk) = imresize(deli(:,:,kk), [size(mri,1) size(mri,2)] ,"nearest");
%         mri_gaussian(:,:,kk) = imgaussfilt(mri(:,:,kk), 0.5, 'FilterSize',3);
%    end
%    mri_filtered = mri(:,:,CNN_slice).*deli_resized(:,:,CNN_slice);
%    mri_filtered(mri_filtered == 0) = nan;
%    mri_filtered_vec = mri_filtered(:);
%    avg = mean(mri_filtered_vec, 'omitnan');
%    sdev = std(mri_filtered_vec, 'omitnan');
%    mri_gaussian_normalized = (mri_gaussian-avg)/sdev;
end