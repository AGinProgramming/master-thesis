function TextFeature = texture_extraction_SLIC(fullfilename, image, textmatlist, voxel_info, labels)

voxel_label = unique(labels);

TextFeature = struct;

for i = 1:length(voxel_label)
    label = unique_labels(i);
    % give current superpixel a mask
    mask = (labels == label);
    
    % extract the area of current superpixel
    image_win = image .* uint8(mask);

    % extracting texture features
    [ROIonly] = prepareVolume(image_win, mask, 'MRscan', voxel_info.PixelSpacing(1), voxel_info.SliceThickness, 1, 'pixelW', 'Global');
    text.Global = getGlobalTextures(ROIonly, 100);
    
    [ROIonly, levels, ~ , ~] = prepareVolume(image_win, mask, 'MRscan', voxel_info.PixelSpacing(1), voxel_info.SliceThickness, 1, 'pixelW', 'Matrix', 'Equal', 64);
    
    glmatrix = getGLCM(ROIonly, levels);
    text.GLCM = getGLCMtextures(glmatrix);
    [ngtmatrix, countvalid] = getNGTDM(ROIonly, levels);
    text.NGTDM = getNGTDMtextures(ngtmatrix, countvalid);
    
    % storing texture features for each superpixel
    for mm = 1:length(textmatlist)
        fieldlist = fieldnames(text.(textmatlist{mm}));
        for ff = 1:length(fieldlist)
            TextFeature.(textmatlist{mm}).(fieldlist{ff})(label) = ...
                text.(textmatlist{mm}).(fieldlist{ff});
        end
    end
end

end