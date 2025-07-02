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

## Read nii file
# input_path = r"G:\NKI dataset\Data_nifti\MRI001\NIFTIs\T2.nii"

# nii_image = nib.load(input_path)

# # ap = argparse.ArgumentParser()
# # ap.add_argument("-i", "--input", required=True, help="Path to the image")
# # args = vars(ap.parse_args())

# # nii_image = nib.load(args["input"])

# image_data = nii_image.get_fdata()
# image_data.ndim == 15
# image_slice = image_data[:, :, image_data.shape[2] // 2]

# image_next = np.rot90(image_slice, 3)

## Read JPEG file
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--input", required=True, help="Path to te image")
# args = vars(ap.parse_args())

# image = img_as_float(skimage.io.imread(args["input"]))

# image_path = r'C:/Users/alexg/Desktop/study/graduation project/code/T2-15.jpeg'
image_path = r'C:/Users/alexg/Desktop/study/graduation project/code/corrected_test.jpeg'
image = skimage.io.imread(image_path)
image_rotate = np.rot90(image, 4)
image_next = image_rotate[:, ::-1]
print(image_next.shape)


# read Excel
filename = 'coordinates.xlsx'
df = pd.read_excel(filename, sheet_name='Sheet1', header=None)
print(df)
# get access to data
minX, minY, maxX, maxY = df.iloc[0, :4]
# region = image_next.crop((minX - 5, minY - 5, maxX + 5, maxY + 5))

region = image_next[minY - 25 : maxY + 10, minX - 20 : maxX + 20]
# clrregion = image_next[minY - 5 : maxY + 5, minX - 5 : maxX + 5]


print(region.shape)
print(f"minX: {minX}, minY: {minY}, maxX: {maxX}, maxY: {maxY}")

segments_1 = slic(region, n_segments=25, compactness=0.01, sigma=1, start_label=1, channel_axis=None)
segments_2 = slic(region, n_segments=25, compactness=0.05, sigma=1, start_label=1, channel_axis=None)
segments_3 = slic(region, n_segments=25, compactness=0.1, sigma=1, start_label=1, channel_axis=None)
segments_4 = slic(region, n_segments=30, compactness=0.01, sigma=1, start_label=1, channel_axis=None)
segments_5 = slic(region, n_segments=30, compactness=0.05, sigma=1, start_label=1, channel_axis=None)

# image = region[:, :, region.shape[2] // 2]

# Find boundaries
boundaries_1 = find_boundaries(segments_1, mode='inner')
boundaries_2 = find_boundaries(segments_2, mode='inner')
boundaries_3 = find_boundaries(segments_3, mode='inner')
boundaries_4 = find_boundaries(segments_4, mode='inner')
boundaries_5 = find_boundaries(segments_5, mode='inner')
print(boundaries_1.shape)
# Overlay boundaries on the original image
image_with_boundaries_1 = mark_boundaries(region, boundaries_1, color=(0, 1, 0))
image_with_boundaries_2 = mark_boundaries(region, boundaries_2, color=(0, 1, 0))
image_with_boundaries_3 = mark_boundaries(region, boundaries_3, color=(0, 1, 0))
image_with_boundaries_4 = mark_boundaries(region, boundaries_4, color=(0, 1, 0))
image_with_boundaries_5 = mark_boundaries(region, boundaries_5, color=(0, 1, 0))



# Display the original image and the segmented image in grayscale
plt.figure(figsize=(12, 6))

plt.subplot(2, 3, 1)
plt.imshow(image_next, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(image_with_boundaries_1, cmap='gray')
plt.title('25 0.05 1')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(image_with_boundaries_2, cmap='gray')
plt.title('25 0.1 1')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(image_with_boundaries_3, cmap='gray')
plt.title('25 0.5 1')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(image_with_boundaries_4, cmap='gray')
plt.title('30 0.1 1')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(image_with_boundaries_5, cmap='gray')
plt.title('30 0.5 1')
plt.axis('off')

plt.show()



image_path = 'corrected_test.jpeg'
image_jpeg = skimage.io.imread(image_path)

nifti_path = r'G:\NKI dataset\Data_nifti\MRI001\NIFTIs\T2.nii'
nifti_image = nib.load(nifti_path)
data_nifti = nifti_image.get_fdata()


# 将JPEG和NIfTI数据转换为一维数组
data_jpeg_flattened = image_jpeg.ravel()
data_nifti_flattened = data_nifti.ravel()

# 绘制直方图
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(data_jpeg_flattened, bins=50, color='blue', alpha=0.7)
plt.title('JPEG Image Histogram')
plt.xlabel('Intensity')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(data_nifti_flattened, bins=50, color='red', alpha=0.7)
plt.title('NIfTI Image Histogram')
plt.xlabel('Intensity')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


