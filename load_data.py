import SimpleITK as sitk
import nibabel as nib
import numpy as np

def data_load_nii(path,return_info=False,return_array=True):
    """
    Parameter
    - path
    - return_info : If True, return will be image,origin,spacing
    - return_array : Generally we return to array.
    """
    image = sitk.ReadImage(path)
   
    if return_info == True:
        spacing = image.GetSpacing()
        origin = image.GetOrigin()

    if return_array == True:
        image = sitk.GetArrayFromImage(image).astype('float32')

    if return_info == True:
        return image,origin,spacing
    else:
        return image

def data_load_dicom(path,return_info=False,return_array=True):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(path)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()

    if return_info == True:
        spacing = image.GetSpacing()
        origin = image.GetOrigin()
    
    if return_array == True:
        image = sitk.GetArrayFromImage(image)
        
    if return_info == True:
        return image,origin,spacing
    else:
        return image
    
def data_save_itk(image, origin, spacing, filename='image.nii.gz'):
    """
    !! Not Completed
    You need need get origin & spacing data.
    Use load_nii(image_path,return_info=True) function to get origin & spacing data.
    """
    image = np.flip(image,1)
    itkimage = sitk.GetImageFromArray(image)
    itkimage.SetSpacing(spacing)
    itkimage.SetOrigin(origin)
#     sitk.WriteImage(itkimage, filename, True) 
    sitk.WriteImage(itkimage, filename, False) 
