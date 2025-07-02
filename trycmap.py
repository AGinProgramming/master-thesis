import matplotlib.pyplot as plt
import numpy as np
import cv2  # Assuming OpenCV is used for image loading

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

nifti_path = r'G:\NKI dataset\Data_nifti\MRI001\NIFTIs\T2.nii'
nifti_image = nib.load(nifti_path)
data_nifti = nifti_image.get_fdata()
data_nifti.ndim == 15
image_slice = data_nifti[:, :, data_nifti.shape[2] // 2]

image_nifti = np.rot90(image_slice, 3)

minni = np.min(image_nifti)
maxni = np.max(image_nifti)
rescaled_image = ((image_nifti - minni) / (maxni - minni)) * (255 - 0)
rescaled_image.astype(np.uint8)

# read Excel
filename = 'coordinates.xlsx'
df = pd.read_excel(filename, sheet_name='Sheet1', header=None)
print(df)
# get access to data
minX, minY, maxX, maxY = df.iloc[0, :4]
# region = image_next.crop((minX - 5, minY - 5, maxX + 5, maxY + 5))

# region = image_next[minY - 25 : maxY + 10, minX - 20 : maxX + 20]
region = rescaled_image[minY - 25 : maxY + 10, minX - 20 : maxX + 20]
# clrregion = image_next[minY - 5 : maxY + 5, minX - 5 : maxX + 5]

print(region.shape)
print(f"minX: {minX}, minY: {minY}, maxX: {maxX}, maxY: {maxY}")

segments_1 = slic(region, n_segments=25, compactness=0.05, sigma=1, start_label=1, channel_axis=None)

boundaries_1 = find_boundaries(segments_1, mode='inner')

image_with_boundaries_1 = mark_boundaries(region, boundaries_1, color=(1, 0, 0))

from skimage.morphology import binary_dilation
dilated_boundaries = binary_dilation(image_with_boundaries_1)

# # Generate segments
# segments_1 = slic(region, n_segments=25, compactness=0.05, sigma=1, start_label=1, channel_axis=None)

# # Find boundaries
# boundaries_1 = find_boundaries(segments_1, mode='inner').astype(np.uint8)

# # Create an empty image for boundaries or use a copy of the original image for overlaying
# boundary_overlay = np.copy(region)

# # Set boundary color. Adjust the RGB values as needed.
# boundary_overlay[boundaries_1, 0] = 255  # R通道
# boundary_overlay[boundaries_1, 1] = 0    # G通道
# boundary_overlay[boundaries_1, 2] = 0    # B通道


# Display the original and boundary-marked images for comparison
plt.figure(figsize=(10, 5))

plt.subplot(2, 4, 1)
plt.imshow(region, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 4, 2)
plt.imshow(image_with_boundaries_1, cmap='gray')
plt.title('gray')
plt.axis('off')

plt.subplot(2, 4, 3)
plt.imshow(image_with_boundaries_1, cmap='binary')
plt.title('binary')
plt.axis('off')

plt.subplot(2, 4, 4)
plt.imshow(image_with_boundaries_1, cmap='hot')
plt.title('hot')
plt.axis('off')

plt.subplot(2, 4, 5)
plt.imshow(image_with_boundaries_1, cmap='cool')
plt.title('cool')
plt.axis('off')

plt.subplot(2, 4, 6)
plt.imshow(image_with_boundaries_1, cmap='bone')
plt.title('bone')
plt.axis('off')

plt.subplot(2, 4, 7)
plt.imshow(image_with_boundaries_1, cmap='copper')
plt.title('copper')
plt.axis('off')

plt.subplot(2, 4, 8)
plt.imshow(image_with_boundaries_1, cmap='jet')
plt.title('jet')
plt.axis('off')

plt.show()