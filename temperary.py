import numpy as np

from skimage.segmentation import slic
import skimage.io
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries, mark_boundaries
from openniftiimages import load_and_display_nifti 

# Replace 'your_image.jpg' with the actual path to your JPEG file
image_path = 'C:/Users/alexg/Desktop/study/graduation project/code/T2.jpeg'

# Read the image using skimage.io.imread
image = skimage.io.imread(image_path)


# Debugging prints
print("Type of nifti_data:", type(image))
print("Shape of nifti_data:", image.shape)

# rotation 90 degree
# image_next = np.rot90(image_rotate)
image_next = np.rot90(image,2)

segments_1 = slic(image_next, n_segments=500, compactness=0.8, sigma=10, start_label=1)

image_next = image_next[:, :, image_next.shape[2] // 2]

boundaries_1 = find_boundaries(segments_1, mode='inner')

image_with_boundaries_1 = mark_boundaries(image_next, boundaries_1)

plt.figure
plt.imshow(image_with_boundaries_1, cmap='gray')
# plt.title('a')
plt.axis('off')
plt.show()
