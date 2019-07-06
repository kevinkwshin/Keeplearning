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
    
def aview_img_to_nii(path_file,path_save):
       
    label = nib.load(path_label[0]).get_data().astype('float32')
    label = np.squeeze(label)
    label = np.fliplr(label)
    labels = nib.Nifti1Image(label, affine=np.eye(4))
    nib.save(labels,path_save)

    return image,label
