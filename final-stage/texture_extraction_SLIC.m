function TextFeature = texture_extraction_SLIC(image, textmatlist, voxel_info, labels, unique_labels)

% voxel_label = unique(labels);

% TextFeature = struct;
%% 思路：labels可以直接获取unique的labels，应该遍历整个图的所有pixels label

% 这里应该改为循环每一个labels中的所有值，与对应unique_labels相交不为0的就给它一个mask


% for i = 1:length(unique_labels)
%     % 获取当前 patient 的 label 矩阵和目标 superpixel 的 label
% %     label_matrix = labels{i};
%     target_superpixel_labels = str2num(unique_labels{i, 2});
          target_superpixel_labels = str2num(unique_labels);

%     disp(['Processing row: ', num2str(i)]);
    disp(['Target superpixel labels: ', mat2str(target_superpixel_labels)]);

    % 遍历解析出的每个 superpixel 标签
    for j = 1:length(target_superpixel_labels)
        target_superpixel_label = target_superpixel_labels(j);
        disp(target_superpixel_label) 


%         % 创建 mask
%         disp(labels)
% 
%         % 生成彩色图像，其中相同标签的像素具有相同的颜色
%         colored_image = label2rgb(labels, 'jet', 'k', 'shuffle');
%         
%         % 显示彩色图像
%         figure;
%         imshow(colored_image);
%         title('Colored Image with Superpixel Labels');
%         
%         % 保存彩色图像
%         save_folder = 'D:\maskc506';
%         if ~exist(save_folder, 'dir')
%             mkdir(save_folder);
%         end
%         save_path = fullfile(save_folder, 'colored_superpixel_image.png');
%         imwrite(colored_image, save_path);
%         
%         disp('Image saved successfully.');

        mask = (labels == target_superpixel_label);
        
%         % 生成彩色图像，其中相同标签的像素具有相同的颜色
%         colored_image_1 = label2rgb(mask, 'jet', 'k', 'shuffle');
%         
%         % 显示彩色图像
%         figure;
%         imshow(colored_image_1);
%         title('Colored Image with Superpixel Labels');
%         
%         % 保存彩色图像
%         save_folder = 'D:\maskc506';
%         if ~exist(save_folder, 'dir')
%             mkdir(save_folder);
%         end
%         save_path = fullfile(save_folder, 'colored_superpixel_image_1.png');
%         imwrite(colored_image_1, save_path);
%         
%         disp('Image saved successfully.');

        
        % 提取当前 superpixel 区域
%         image_SLIC = image .* uint8(mask);
        image_SLIC = image .* mask;

        disp(class(image_SLIC))
        disp(class(mask))
        disp(size(mask))

        % 提取纹理特征
        [ROIonly] = prepareVolume(image_SLIC, mask, 'MRscan', voxel_info.PixelSpacing(1), voxel_info.SliceThickness, 1, 'pixelW', 'Global');
        text.Global = getGlobalTextures(ROIonly, 100);

        [ROIonly, levels, ~ , ~] = prepareVolume(image_SLIC, mask, 'MRscan', voxel_info.PixelSpacing(1), voxel_info.SliceThickness, 1, 'pixelW', 'Matrix', 'Equal', 64);

        glmatrix = getGLCM(ROIonly, levels);
        text.GLCM = getGLCMtextures(glmatrix);
        [ngtmatrix, countvalid] = getNGTDM(ROIonly, levels);
        text.NGTDM = getNGTDMtextures(ngtmatrix, countvalid);
        glszmatrix = getGLSZM(ROIonly,levels);
        text.GLSZM = getGLSZMtextures(glszmatrix);
        glrlmatrix = getGLRLM(ROIonly,levels);
        text.GLRLM = getGLRLMtextures(glrlmatrix);

        % 存储每个 superpixel 的纹理特征
        for mm = 1:length(textmatlist)
            fieldlist = fieldnames(text.(textmatlist{mm}));
            for ff = 1:length(fieldlist)
                % 使用合适的索引（如 i 和 j 结合）
                TextFeature.(textmatlist{mm}).(fieldlist{ff})(j) = ...
                    text.(textmatlist{mm}).(fieldlist{ff});
            end
        end
    end
% end
% for mm=1:length(textmatlist)
%     fieldlist = fieldnames(text.(textmatlist{mm}));
%     for ff = 1:length(fieldlist)
%         
%         TextFeature.(textmatlist{mm}).(fieldlist{ff})= ...
%             conv2(TextFeature.(textmatlist{mm}).(fieldlist{ff}),H,'same');
%         TextFeature.(textmatlist{mm}).(fieldlist{ff})(end:d1,end:d2)=0;
%     end
% end

TextFeature.grayscale.raw =image;
end
    
%     % 创建 mask
%     mask = (label_matrix == target_superpixel_label);
%     
%     % 提取当前 superpixel 区域
%     image_SLIC = image .* uint8(mask);
% 
%     % extracting texture features
%     [ROIonly] = prepareVolume(image_SLIC, mask, 'MRscan', voxel_info.PixelSpacing(1), voxel_info.SliceThickness, 1, 'pixelW', 'Global');
%     text.Global = getGlobalTextures(ROIonly, 100);
%     
%     [ROIonly, levels, ~ , ~] = prepareVolume(image_SLIC, mask, 'MRscan', voxel_info.PixelSpacing(1), voxel_info.SliceThickness, 1, 'pixelW', 'Matrix', 'Equal', 64);
%     
%     glmatrix = getGLCM(ROIonly, levels);
%     text.GLCM = getGLCMtextures(glmatrix);
%     [ngtmatrix, countvalid] = getNGTDM(ROIonly, levels);
%     text.NGTDM = getNGTDMtextures(ngtmatrix, countvalid);
%     
%     % storing texture features for each superpixel
%     for mm = 1:length(textmatlist)
%         fieldlist = fieldnames(text.(textmatlist{mm}));
%         for ff = 1:length(fieldlist)
%             TextFeature.(textmatlist{mm}).(fieldlist{ff})(label) = ...
%                 text.(textmatlist{mm}).(fieldlist{ff});
%         end
%     end
% end
% 
% end