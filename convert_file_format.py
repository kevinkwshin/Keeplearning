def convert_aview_img_to_nii(path_file,path_save):  # Aview label to nifti label
    label = nib.load(path_file).get_data().astype('float32')
    label = np.squeeze(label)
    label = np.fliplr(label)
    label = nib.Nifti1Image(label, affine=np.eye(4))
    nib.save(label,path_save)

def convert_dicom_to_nii(path_file,path_save):
    
def convert_dicom_to_png(path_file,path_save):
