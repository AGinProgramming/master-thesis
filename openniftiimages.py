import os
import numpy as np
from PIL import Image

from nibabel.testing import data_path

def load_and_display_nifti(file_path):
    # example_filename = os.path.join(data_path, 'G:/NKI dataset/Data_nifti/MRI001/NIFTIs/ADC.nii')
    # example_filename = os.path.join(data_path, 'G:/NKI dataset/Data_nifti/MRI001/NIFTIs/ADC_v3.nii')
    example_filename = os.path.join(data_path, file_path)
    # start with ADC_v3.nii and T2.nii

    import nibabel as nib
    img = nib.load(example_filename)

    print(img.shape)

    img.get_data_dtype() == np.dtype(np.int16)

    print(img.affine.shape)

    hdr = img.header
    hdr.get_xyzt_units()

    raw = hdr.structarr
    # raw['xyzt_units']

    # import numpy as np
    # data = np.ones((32, 32, 15, 100), dtype=np.int16)
    # img = nib.Nifti1Image(data, np.eye(4))
    # img.get_data_dtype() == np.dtype(np.int16)

    # img.header.get_xyzt_units()

    # nib.save(img, os.path.join('newimg', 'MRI001newADC'))  

    import matplotlib.pyplot as plt

    data = img.get_fdata()
    print(data.shape)
    # type(data)

    #slice_to_display = data.shape[2] // 2
    slice_to_display = 15
    slice_data = data[:, :, slice_to_display]

    # rotation 90 degree
    rotated_slice_data = np.rot90(slice_data, 3)
    print(rotated_slice_data.shape)
    # image = Image.fromarray(rotated_slice_data)
    # # print(image.shape)
    # # image.save()
    # image = image.convert("L")
    # image.save("test.jpeg", )
    


    # 假设rotated_slice_data是你的图像数据
    # 归一化图像数据到0-255
    normalized_data = (rotated_slice_data - np.min(rotated_slice_data)) / (np.max(rotated_slice_data) - np.min(rotated_slice_data)) * 255

    # 转换为uint8
    normalized_data = normalized_data.astype(np.uint8)

    # 从归一化后的数据创建图像
    image = Image.fromarray(normalized_data).convert("L")

    # 保存图像
    image.save("corrected_test.jpeg")








    # image.save('T2test.jpeg')
    # inverse rotation 90 degree
    # rotated_slice_data = np.rot90(slice_data, k=-1)

    # plot the slice
    plt.imshow(rotated_slice_data, cmap='gray')
    # plt.title('MRI Slice')
    # plt.colorbar()
    plt.axis('off')
    # plt.savefig('T2_15.jpg', format='jpeg')
    # plt.tight_layout()

    file_name_pic = 'T2-15'
    plt.savefig(f'{file_name_pic}', bbox_inches='tight', pad_inches=0, format='jpeg')
    plt.show()
    plt.close()

    return data  # Return the NIfTI data array

if __name__ == "__main__":
    from nibabel.testing import data_path
    file_path = r'G:\NKI dataset\Data_nifti\MRI001\NIFTIs\T2.nii'
    nifti_data = load_and_display_nifti(file_path)
    print("Type of nifti_data:", type(nifti_data))
    print("Shape of nifti_data:", nifti_data.shape)

