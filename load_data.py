import SimpleITK as sitk
import nibabel as nib

def load_nii(path,return_info=False,return_array=True):
    """
    Parameter
    - path
    - return_info : If True, return will be image,spacing,origin
    - return_array : Generally we return to array.
    """
    image = sitk.ReadImage(path)
   
    if return_info == True:
        spacing = image.GetSpacing()
        origin = image.GetOrigin()

    if return_array == True:
        image = sitk.GetArrayFromImage(image).astype('float32')

    if return_info == True:
        return image,spacing,origin
    else:
        return image

def load_dicom(path,return_info=False,return_array=True):
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
        return image,spacing,origin
    else:
        return image
