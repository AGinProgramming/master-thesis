import os
import scipy
import numpy as np
import pandas as pd
import nibabel as nib
from PIL import Image
from skimage import measure
import matplotlib.pyplot as plt
from skimage.color import label2rgb
from skimage.util import img_as_float
from skimage.segmentation import slic
from scipy.spatial.distance import cdist
from sklearn.metrics import jaccard_score
from skimage.measure import label, regionprops
from openniftiimages import load_and_display_nifti
from scipy.spatial.distance import directed_hausdorff
from skimage.segmentation import find_boundaries, mark_boundaries


patient_info_file = r'G:\Annekoos Schaap - code\Data\All_imaging_features_NKI.xlsx'
base_nifti_path = r'G:\NKI dataset\Data_nifti'

# Read the patient info
patient_info_df = pd.read_excel(patient_info_file)

# read Excel
filename = 'coordinates.xlsx'
df = pd.read_excel(filename, sheet_name='Sheet1', header=None)
print(df)

print(len(df))
print(len(patient_info_df))

segment = []
mask_corruption = []
# mask_prediction = []

for index, row in patient_info_df.iterrows():
    patient_name = row['Patient name ']
    slice_no = int(np.floor(row['Slice no. ']))  # Assuming 'Slice no.' column contains integers
    print(f"Processing {patient_name}, Slice {slice_no}...")
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

        # print(region.shape)
        # print(f"minX: {minX}, minY: {minY}, maxX: {maxX}, maxY: {maxY}")

        # segments = slic(region, n_segments=round(region.shape[0]*region.shape[1]/1000), compactness=0.1, sigma=1, start_label=1, channel_axis=None)
        segments = slic(region, n_segments=30, compactness=0.05, sigma=1, start_label=1, channel_axis=None)
        boundaries_1 = find_boundaries(segments, mode='inner')
        image_with_boundaries_1 = mark_boundaries(region, boundaries_1, color=(1, 0, 0))

        mat_file_path = os.path.join(base_nifti_path, 'logic_adjusted_masks', f'adjusted_patient_{patient_name}', f'adjusted_prostate_slice_{patient_name}', f'adjusted_prostate_slice_{slice_no}.mat')
        mat = scipy.io.loadmat(mat_file_path)
        prostate_mask = mat['logic_image']
        prostate_mask = prostate_mask.astype(bool)
        prostate_mask = prostate_mask[y_min:y_max, x_min:x_max]
        print(prostate_mask.shape)
        # predicted_mask = np.zeros_like(prostate_mask)

        prostate_boundaries = find_boundaries(prostate_mask, mode='inner')

        plt.figure(figsize=(10, 10))
        plt.imshow(region, cmap='gray')
        plt.imshow(label2rgb(boundaries_1, bg_label=0, colors=['red'], bg_color=None, alpha=0.3))
        plt.imshow(label2rgb(prostate_boundaries, bg_label=0, colors=['blue'], bg_color=None, alpha=0.3))
        plt.axis('off')

        save_folder = r'C:\Users\alexg\Desktop\study\graduation project\code\mask_comparison'
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, f'patient_{patient_name}_comparison.png')

        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        save_path

        segment.append(segments)

        mask_corruption.append(prostate_mask)
        # mask_prediction.append(predicted_mask)

    print(f"Finished processing {patient_name}, Slice {slice_no}.\n")
    
print("All patients processed. Calculating metrics...")


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

    return dice_score, dice_score_from_jaccard, hausdorff_dist

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

    # print(points1.shape)
    # print(points2.shape)
    if points1.size == 0 or points2.size == 0:
        return float('inf')  
            
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

# for index, row in patient_info_df.iterrows():
#     patient_name = row['Patient name ']

#     dice_score, dice_score_from_jaccard, hausdorff_dist = calculate_metrics(segment, mask_corruption, threshold)
#     asd = calculate_asd(segment, mask_corruption)
    
#     # 输出当前分割的指标，并包括病人的编号
#     print(f'Patient {index} - {patient_name}:')
#     print(f'  Dice Score: {dice_score}')
#     print(f'  Dice Score from Jaccard Index: {dice_score_from_jaccard}')
#     print(f'  Hausdorff Distance: {hausdorff_dist}')
#     print(f'  Average Surface Distance: {asd}\n')


patient_No = patient_info_df['Patient name '].tolist()
print(patient_No)

table = []

print(segment)

# for i, (segment, mask_corruption, mask_prediction) in enumerate(zip(segment, mask_corruption, mask_prediction), start=1):
#     dice_score, dice_score_from_jaccard, hausdorff_dist, asd = calculate_metrics(segment, prostate_mask, threshold)
for i, (segment, mask_corruption,patient_No) in enumerate(zip(segment, mask_corruption,patient_No), start=1):
    dice_score, dice_score_from_jaccard, hausdorff_dist = calculate_metrics(segment, mask_corruption, threshold)
    asd = calculate_asd(segment, mask_corruption)

    table.append({
        'Patient_No': patient_No,
        'Dice_Score': dice_score,
        'Dice_Score_from_Jaccard': dice_score_from_jaccard,
        'Hausdorff_Distance': hausdorff_dist,
        'ASD': asd
    })
    
    # Output the metrics for the current segmentation
    print(f'Patient {patient_No}:')
    print(f'Segmentation {i}:')
    print(f'  Dice Score: {dice_score}')
    print(f'  Dice Score from Jaccard Index: {dice_score_from_jaccard}')
    print(f'  Hausdorff Distance: {hausdorff_dist}')
    print(f'  Average Surface Distance: {asd}\n')


# 使用pandas创建DataFrame
evaluation_save = pd.DataFrame(table)

# 指定要写入的Excel文件路径
evaluation_save_path = r'C:\Users\alexg\Desktop\study\graduation project\code\table2.xlsx'

# 将DataFrame写入Excel文件，不保存行索引
evaluation_save.to_excel(evaluation_save_path, index=False)

print('Data saved to Excel file successfully.')
