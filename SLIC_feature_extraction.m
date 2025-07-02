function TextFeature = texture_extraction_SLIC(fullfilename, image, textmatlist, voxel_info, labels)
% -------------------------------------------------------------------------
% function [textures] =  TextFeature = texture_extraction_movwin(Projections, fullfilename, textmatlist, imagelist, winsize)
% -------------------------------------------------------------------------
% DESCRIPTION: function to extract texture features by a moving window
% -------------------------------------------------------------------------

% imagelist = different projections/frames of the CEUS (peak image, maximum intensity projection,
% wash-in, mean intensity projections ecc..)
% Ng = number of gray levels

% applied a spatial convolution to get the obtained parametric maps back to
% the original size, despite using a stride of 3

%% Texture map

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

























% fprintf('\n\n Processing Pat %i \n',fullfilename);
% image = input;
% [d1,d2] = size(image);
Ng = 64; % grey levels

%filter to upsample

% win_h = winsize;
% win_w = winsize;
%roisize = [win_h, win_w];
% vert_shift =(win_h-1)/2;
% horiz_shift = (win_w-1)/2;
% stride = 3;
% sz = stride*stride;

% H = fspecial('average',[sz sz]);

for m=(win_h-1)/2+1:stride:d1-(win_h-1)/2
    for n=(win_w-1)/2+1:stride:d2-(win_w-1)/2
        
        mask =zeros(d1,d2);
        mask(m-vert_shift:m+vert_shift, n-horiz_shift:n+horiz_shift)=1;
        
        image_win = image(logical(mask));
        image_win = reshape(image_win, [win_h,win_w]);
        
        %prepareVolume(volume,mask,scanType,pixelW,sliceS,R,scale,textType,quantAlgo,Ng)
        % used 1st time (below)
        %[ROIonly] = prepareVolume(image_win,ones(size(image_win)),'MRscan',0.024,1,1,'pixelW','Global');
        % used 2nd time (below)
        %[ROIonly] = prepareVolume(image_win,ones(size(image_win)),'MRscan',1,1,1,'pixelW','Global');
        [ROIonly] = prepareVolume(image_win,ones(size(image_win)),'MRscan',voxel_info.PixelSpacing(1),voxel_info.SliceThickness,1,'pixelW','Global');
        text.Global = getGlobalTextures(ROIonly,100);
        
        [ROIonly,levels,~ ,~] = prepareVolume(image_win,ones(size(image_win)),'MRscan',voxel_info.PixelSpacing(1),voxel_info.SliceThickness,1,'pixelW','Matrix','Equal',Ng);
        %[ROIonly,levels,~ ,~] = prepareVolume(image_win,ones(size(image_win)),'MRscan',infoDCE.PixelSpacing(1),1,1,1,'Matrix','Equal',Ng);
        glmatrix = getGLCM(ROIonly,levels);
        text.GLCM = getGLCMtextures(glmatrix);
        [ngtmatrix, countvalid] = getNGTDM(ROIonly,levels);
        text.NGTDM = getNGTDMtextures(ngtmatrix, countvalid);
        glszmatrix = getGLSZM(ROIonly,levels);
        text.GLSZM = getGLSZMtextures(glszmatrix);
        glrlmatrix = getGLRLM(ROIonly,levels);
        text.GLRLM = getGLRLMtextures(glrlmatrix);
        
        for mm=1:length(textmatlist)
            fieldlist = fieldnames(text.(textmatlist{mm}));
            for ff = 1:length(fieldlist)
                TextFeature.(textmatlist{mm}).(fieldlist{ff})(m,n) = ...
                    text.(textmatlist{mm}).(fieldlist{ff});
            end
            %             fprintf('Pat %s, %s analysis done',fullfilename,textmatlist{mm});
        end
    end
end

for mm=1:length(textmatlist)
    fieldlist = fieldnames(text.(textmatlist{mm}));
    for ff = 1:length(fieldlist)
        
        TextFeature.(textmatlist{mm}).(fieldlist{ff})= ...
            conv2(TextFeature.(textmatlist{mm}).(fieldlist{ff}),H,'same');
        TextFeature.(textmatlist{mm}).(fieldlist{ff})(end:d1,end:d2)=0;
    end
end

TextFeature.grayscale.raw =image;
% fprintf('\n\n Pat %i done\n',fullfilename);
% fffname = strcat('C:\Users\320106846\Documents\Cancer Research\Prostate Cancer research\MSc and BEP projects\T2w_features\Pat',num2str(fullfilename),'_TextFeatures_T2w','_slice',num2str(ind),'_window',num2str(winsize));
% if isfile(fffname)
% else
%     save(fffname,'TextFeature')
% end

end