import SimpleITK as sitk
import nibabel as nib

def load_nii(path)    
    image = sitk.GetArrayFromImage(sitk.ReadImage(x_list)).astype('float32')
    return image

def load_dicom(path):    
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(path)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    image = sitk.GetArrayFromImage(image)
    return image
