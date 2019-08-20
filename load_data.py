import SimpleITK as sitk
import nibabel as nib

def load_nii(path,return_array=True):
    image = sitk.ReadImage(path)
    if return_array == True:
        image = sitk.GetArrayFromImage(image).astype('float32')
    return image

def load_dicom(path,return_array=True):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(path)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    if return_array == True:
        image = sitk.GetArrayFromImage(image)
    return image
