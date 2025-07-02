
import numpy as np
import pandas as pd

from skimage.segmentation import slic
import skimage.io
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries, mark_boundaries
from openniftiimages import load_and_display_nifti 
from skimage.util import img_as_float
import argparse
import nibabel as nib
import skimage.transform
from PIL import Image

import nibabel as nib
import os


# image_path = r'C:/Users/alexg/Desktop/study/graduation project/code/T2-15.jpeg'
# image_path = r'C:/Users/alexg/Desktop/study/graduation project/code/test.jpeg'
# image_path = 'corrected_test.jpeg'
# image = skimage.io.imread(image_path)
# image_rotate = np.rot90(image, 4)
# image_next = image_rotate[:, ::-1]
# print(image_next.shape)









# # read the MRIs
# excel_path = 'MRI_acquisition_settings_T2_DCE_hubert.xlsx'
# df = pd.read_excel(excel_path, sheet_name='T2')

# # base path for all the nifti data
# base_path = r'G:\\NKI dataset\\Data_nifti'

# for index, row in df.iterrows():
#     folder_name = row['Patient']
#     slice_number = row['Slice']  # where can I find this Slice data??????
#     nifti_path = os.path.join(base_path, folder_name, 'NIFTIs', 'T2.nii')
#     print(nifti_path)

#     nifti_image = nib.load(nifti_path)
#     data_nifti = nifti_image.get_fdata()
#     image_slice = data_nifti[:, :, slice_number]
#     image_nifti = np.rot90(image_slice, 3)

#     # connect to the rest of the procedures ... ...
#     minni = np.min(image_nifti)
#     maxni = np.max(image_nifti)
#     rescaled_image = ((image_nifti - minni) / (maxni - minni)) * (255 - 0)
#     rescaled_image.astype(np.uint8)

#     data_nifti_flattened = rescaled_image.ravel()

# plt.figure(figsize=(10, 4))
# plt.hist(data_nifti_flattened, bins=50, color='red', alpha=0.7)
# plt.title('NIfTI Image Histogram')
# plt.xlabel('Intensity')
# plt.ylabel('Frequency')

# plt.tight_layout()
# plt.show()








# nifti_path = r'G:\NKI dataset\Data_nifti\MRI001\NIFTIs\T2.nii'
# nifti_image = nib.load(nifti_path)
# data_nifti = nifti_image.get_fdata()
# data_nifti.ndim == 15
# image_slice = data_nifti[:, :, data_nifti.shape[2] // 2]

# image_nifti = np.rot90(image_slice, 3)

# minni = np.min(image_nifti)
# maxni = np.max(image_nifti)
# rescaled_image = ((image_nifti - minni) / (maxni - minni)) * (255 - 0)
# rescaled_image.astype(np.uint8)


# # 将JPEG和NIfTI数据转换为一维数组
# data_jpeg_flattened = image_next.ravel()
# data_nifti_flattened = rescaled_image.ravel()

# # 绘制直方图
# plt.figure(figsize=(10, 4))
# plt.subplot(1, 2, 1)
# plt.hist(data_jpeg_flattened, bins=50, color='blue', alpha=0.7)
# plt.title('JPEG Image Histogram')
# plt.xlabel('Intensity')
# plt.ylabel('Frequency')

# plt.subplot(1, 2, 2)
# plt.hist(data_nifti_flattened, bins=50, color='red', alpha=0.7)
# plt.title('NIfTI Image Histogram')
# plt.xlabel('Intensity')
# plt.ylabel('Frequency')

# plt.tight_layout()
# plt.show()




patient_info_file = r'G:\Annekoos Schaap - code\Data\All_imaging_features_NKI.xlsx'
base_nifti_path = r'G:\NKI dataset\Data_nifti'

# Read the patient info
patient_info_df = pd.read_excel(patient_info_file)

# read Excel
filename = 'coordinates.xlsx'
df = pd.read_excel(filename, sheet_name='Sheet1', header=None)
print(df)

for index, row in patient_info_df.iterrows():
    patient_name = row['Patient name']
    slice_no = int(np.floor(row['Slice no.']))  # Assuming 'Slice no.' column contains integers
    # Construct the path to the NIfTI file
    nifti_path = os.path.join(base_nifti_path, patient_name, 'NIFTIs', 'T2.nii')
    
    # Load the NIfTI image
    nifti_image = nib.load(nifti_path)
    data_nifti = nifti_image.get_fdata()
    
    # Check if the number of dimensions is correct
    if data_nifti.ndim == 3:
        # Get the specific slice
        image_slice = data_nifti[:, :, slice_no]
        
        # Process the image slice
        image_nifti = np.rot90(image_slice, 3)
        minni = np.min(image_nifti)
        maxni = np.max(image_nifti)
        rescaled_image = ((image_nifti - minni) / (maxni - minni)) * 255
        rescaled_image = rescaled_image.astype(np.uint8)

        # Get coordinates for the current patient
        # Assuming each patient has a corresponding row in the coordinates Excel file
        minX, minY, maxX, maxY = df.iloc[index, :4]
        
        # Crop the region from the rescaled image
        # Ensure that the indices do not go out of bounds
        y_min = max(minY - 25, 0)
        y_max = min(maxY + 10, rescaled_image.shape[0])
        x_min = max(minX - 20, 0)
        x_max = min(maxX + 20, rescaled_image.shape[1])
        region = rescaled_image[y_min:y_max, x_min:x_max]

        print(region.shape)
        print(f"minX: {minX}, minY: {minY}, maxX: {maxX}, maxY: {maxY}")

        segments = slic(region, n_segments=round(region.shape(0)*region.shape(1)/1000), compactness=0.1, sigma=1, start_label=1, channel_axis=None)
        boundaries_1 = find_boundaries(segments, mode='inner')
        image_with_boundaries_1 = mark_boundaries(region, boundaries_1, color=(1, 0, 0))

        mat_file_path = os.path.join(base_nifti_path, 'logic_adjusted_masks', f'adjusted_patient_{patient_name}', f'adjusted_prostate_slice_{patient_name}', f'adjusted_prostate_slice_{slice_no}.mat')
        mat = scipy.io.loadmat(mat_file_path)
        prostate_mask = mat['logic_image']
        prostate_mask = prostate_mask.astype(bool)
        prostate_mask = prostate_mask[y_min:y_max, x_min:x_max]









# # get access to data
# minX, minY, maxX, maxY = df.iloc[0, :4]
# # region = image_next.crop((minX - 5, minY - 5, maxX + 5, maxY + 5))

# # region = image_next[minY - 25 : maxY + 10, minX - 20 : maxX + 20]
# region = rescaled_image[minY - 25 : maxY + 10, minX - 20 : maxX + 20]
# # clrregion = image_next[minY - 5 : maxY + 5, minX - 5 : maxX + 5]

# print(region.shape)
# print(f"minX: {minX}, minY: {minY}, maxX: {maxX}, maxY: {maxY}")

# from skimage import exposure
# region = exposure.equalize_hist(region) * 255

segments_1 = slic(region, n_segments=25, compactness=0.05, sigma=1, start_label=1, channel_axis=None)
segments_2 = slic(region, n_segments=25, compactness=0.1, sigma=1, start_label=1, channel_axis=None)
segments_3 = slic(region, n_segments=25, compactness=0.15, sigma=1, start_label=1, channel_axis=None)
segments_4 = slic(region, n_segments=30, compactness=0.05, sigma=1, start_label=1, channel_axis=None)
segments_5 = slic(region, n_segments=30, compactness=0.1, sigma=1, start_label=1, channel_axis=None)

# image = region[:, :, region.shape[2] // 2]

# Find boundaries
boundaries_1 = find_boundaries(segments_1, mode='inner')
boundaries_2 = find_boundaries(segments_2, mode='inner')
boundaries_3 = find_boundaries(segments_3, mode='inner')
boundaries_4 = find_boundaries(segments_4, mode='inner')
boundaries_5 = find_boundaries(segments_5, mode='inner')
print(boundaries_1.shape)
# Overlay boundaries on the original image
image_with_boundaries_1 = mark_boundaries(region, boundaries_1, color=(1, 0, 0))
image_with_boundaries_2 = mark_boundaries(region, boundaries_2, color=(1, 0, 0))
image_with_boundaries_3 = mark_boundaries(region, boundaries_3, color=(1, 0, 0))
image_with_boundaries_4 = mark_boundaries(region, boundaries_4, color=(1, 0, 0))
image_with_boundaries_5 = mark_boundaries(region, boundaries_5, color=(1, 0, 0))


# Display the original image and the segmented image in grayscale
plt.figure(figsize=(12, 6))

plt.subplot(2, 3, 1)
plt.imshow(rescaled_image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(image_with_boundaries_1, cmap='viridis')
plt.title('25 0.01 1')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(image_with_boundaries_2, cmap='gray')
plt.title('25 0.05 1')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(image_with_boundaries_3, cmap='gray')
plt.title('25 0.1 1')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(image_with_boundaries_4, cmap='gray')
plt.title('30 0.05 1')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(image_with_boundaries_5, cmap='gray')
plt.title('30 0.1 1')
plt.axis('off')

plt.show()


import scipy
from skimage.measure import label, regionprops
from scipy.spatial.distance import directed_hausdorff
from sklearn.metrics import jaccard_score 

# Load the prostate mask
mat = scipy.io.loadmat(r'G:\NKI dataset\Data_nifti\logic_adjusted_masks\adjusted_patient_MRI001\adjusted_prostate_slice_MRI001\adjusted_prostate_slice_15.mat')
prostate_mask = mat['logic_image']
prostate_mask = prostate_mask.astype(bool)
prostate_mask = prostate_mask[minY - 25 : maxY + 10, minX - 20 : maxX + 20]

# Initialize an empty image for the predicted mask
predicted_mask = np.zeros_like(prostate_mask)

# Define the threshold
threshold = 0.75 

def calculate_metrics(segments, prostate_mask, threshold):
    # Initialize an empty image for the predicted mask
    predicted_mask = np.zeros_like(prostate_mask, dtype=int)

    # For each superpixel, calculate its overlap with the prostate mask
    for region in regionprops(label(segments)):
        # Extract the superpixel region
        superpixel = (segments == region.label)
        
        # Calculate overlap with the prostate mask
        overlap = np.sum(superpixel & prostate_mask) / np.sum(superpixel)
        
        # If overlap exceeds the threshold, consider it as part of the prostate
        if overlap > threshold:
            predicted_mask[superpixel] = 1

    
    # Dice score 
    dice_score = 2 * np.sum(predicted_mask & prostate_mask) / (np.sum(predicted_mask) + np.sum(prostate_mask))

    # Jaccard index Dice score: 2 * Jaccard index / (1 + Jaccard index)
    jaccard_index = jaccard_score(prostate_mask.flatten(), predicted_mask.flatten())
    dice_score_from_jaccard = 2 * jaccard_index / (1 + jaccard_index)

    # Hausdorff distance
    u = np.transpose(np.nonzero(prostate_mask))
    v = np.transpose(np.nonzero(predicted_mask))
    hausdorff_dist = max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0])

    print(prostate_mask.shape)
    print(predicted_mask.shape)



    # 提取两个掩模的边界点
    points1 = extract_boundary_points(prostate_mask)
    points2 = extract_boundary_points(predicted_mask)

    print(points1.shape)
    print(points2.shape)
    
    # 计算点集之间的距离
    distances_1_to_2 = cdist(points1, points2, metric='euclidean')
    distances_2_to_1 = cdist(points2, points1, metric='euclidean')
    
    # 找到最短距离并计算平均值
    min_distances_1_to_2 = np.min(distances_1_to_2, axis=1)
    min_distances_2_to_1 = np.min(distances_2_to_1, axis=1)
    
    # 计算两个方向的平均表面距离
    asd_1_to_2 = np.mean(min_distances_1_to_2)
    asd_2_to_1 = np.mean(min_distances_2_to_1)
    
    # 返回双向平均表面距离
    asd = (asd_1_to_2 + asd_2_to_1) / 2

    return dice_score, dice_score_from_jaccard, hausdorff_dist, asd

from skimage import measure
from scipy.spatial.distance import cdist

def extract_boundary_points(mask):
    # 提取边界点
    contours = measure.find_contours(mask, level=0.5)
    points = []
    for contour in contours:
        for point in contour:
            points.append(point)
    
    return np.array(points)

def calculate_asd(mask1, mask2):
    # 提取两个掩模的边界点
    points1 = extract_boundary_points(mask1)
    points2 = extract_boundary_points(mask2)

    print(points1.shape)
    print(points2.shape)
    
    # 计算点集之间的距离
    distances_1_to_2 = cdist(points1, points2, metric='euclidean')
    distances_2_to_1 = cdist(points2, points1, metric='euclidean')
    
    # 找到最短距离并计算平均值
    min_distances_1_to_2 = np.min(distances_1_to_2, axis=1)
    min_distances_2_to_1 = np.min(distances_2_to_1, axis=1)
    
    # 计算两个方向的平均表面距离
    asd_1_to_2 = np.mean(min_distances_1_to_2)
    asd_2_to_1 = np.mean(min_distances_2_to_1)
    
    # 返回双向平均表面距离
    return (asd_1_to_2 + asd_2_to_1) / 2

# # 调用函数计算ASD
# asd = calculate_asd(prostate_mask, predicted_mask)


segments = [segments_1, segments_2, segments_3, segments_4, segments_5]

# Loop through each segmentation and calculate metrics
for i, segment in enumerate(segments, start=1):
    dice_score, dice_score_from_jaccard, hausdorff_dist, asd = calculate_metrics(segment, prostate_mask, threshold)
    
    # Output the metrics for the current segmentation
    print(f'Segmentation {i}:')
    print(f'  Dice Score: {dice_score}')
    print(f'  Dice Score from Jaccard Index: {dice_score_from_jaccard}')
    print(f'  Hausdorff Distance: {hausdorff_dist}')
    print(f'  Average Surface Distance: {asd}\n')
