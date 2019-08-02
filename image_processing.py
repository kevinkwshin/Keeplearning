import numpy as np
import SimpleITK as sitk
import nibabel as nib
from scipy import ndimage

def image_normalize(self,slice):
     '''
         input: unnormalized slice 
         OUTPUT: normalized clipped slice
     '''
     b = np.percentile(slice, 99)
     t = np.percentile(slice, 1)
     slice = np.clip(slice, t, b)
     image_nonzero = slice[np.nonzero(slice)]
     if np.std(slice)==0 or np.std(image_nonzero) == 0:
         return slice
     else:
         tmp= (slice - np.mean(image_nonzero)) / np.std(image_nonzero)
         #since the range of intensities is between 0 and 5000 ,the min in the normalized slice corresponds to 0 intensity in unnormalized slice
         #the min is replaced with -9 just to keep track of 0 intensities so that we can discard those intensities afterwards when sampling random patches
         tmp[tmp==tmp.min()]=-9
         return tmp

def z_normalization(img, num_channels):
    for i in range(num_channels):
        img[..., i] -= np.mean(img[..., i])
        img[..., i] /= np.std(img[..., i])
    return img

def sample_z_norm(data, mean=0.174634420286961, sd=0.11619528340846214):
    data -= mean
    data /= sd
    return data

def image_preprocess_float(img):
# For CT
#     img[img < -1024] = -1024.
#     img[img >= 3071] = 3071.
#     img += 1024.
#     img /= (2**12-1)
    
    b = np.percentile(img, 99)
    t = np.percentile(img, 1)
    img = np.clip(img, t, b)
    img= (img - b) / (t-b)
    img= 1-img
    return img

def image_preprocess_CT_uint8(img):
    img[img < -1024] = -1024.
    img[img >= 3071] = 3071.
    img += 1024.
    img /= (2**12-1)
    return img

def image_resize(data, img_dep=200., img_rows=200., img_cols=200.): # 3D image
    # mode : 'constant’ for image, ‘nearest' for mask
    resize_factor = (img_dep/data.shape[0], img_rows/data.shape[1], img_cols/data.shape[2])
    data = ndimage.zoom(data, resize_factor, order=0, mode='constant', cval=0.0)
    return data

def image_windowing(img, ww=1800, wl=400):
    # preprocessing for CT image (medical)
    # img shape [width, height, depth]
    # ww & wl: bone preset
    maxp = np.max(img)
    minp = np.min(img)

    a = wl - (ww/2)
    b = wl + (ww/2)
    slope = (maxp - minp)/ww
    intercept = maxp - (slope*b)

    img[img < a] = minp
    img[img > b] = maxp
    img = np.where((img >= a) & (img <= b),np.round(slope*img + intercept), img)

    return img

def image_save_nii(data,path):
    # input shape  : (data)
    # output shape : (data)
    
    data_nii = np.transpose(data)
    output = nib.Nifti1Image(data_nii, affine=np.eye(4))
    nib.save(output, os.path.join(path))
