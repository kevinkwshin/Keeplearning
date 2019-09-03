import torch
import numpy as np
import SimpleITK as sitk
import nibabel as nib
from scipy import ndimage
from skimage import measure

def image_normalize(self,slice):
     """
         input: unnormalized slice 
         OUTPUT: normalized clipped slice
     """
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
    """
    z_normalization
    """
    for i in range(num_channels):
        img[..., i] -= np.mean(img[..., i])
        img[..., i] /= np.std(img[..., i])
    return img

def sample_z_norm(data, mean=0.174634420286961, sd=0.11619528340846214):
    data -= mean
    data /= sd
    return data

# def image_preprocess_float(img):
#     """
#     Convert image array into 0~1 float
#     """
#     b = np.percentile(img, 99)
#     t = np.percentile(img, 1)
#     img = np.clip(img, t, b)
#     img= (img - b) / (t-b)
#     img= 1-img
#     return img

def image_preprocess_float(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)    
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x

def image_preprocess_uint8(img):
    """
    Cut off & Convert image array into 0~255
    """
    b = np.percentile(img, 99)
    t = np.percentile(img, 1)
    img = np.clip(img, t, b)
    img= (img - b) / (t-b)
    img= np.round((1-img)*255)
    return img

def image_preprocess_CT_uint8(img):
    """
    Convert CT image array in HU into 0~255 uint
    """
    img[img < -1024] = -1024.
    img[img >= 3071] = 3071.
    img += 1024.
    img /= (2**12-1)
    return img

def image_resize_3D(data, img_dep=200., img_rows=200., img_cols=200.,mode='constant'): # 3D image
    """
    Resizer for 3D image
    mode : 'constant’ for image, ‘nearest' for mask
    """
    resize_factor = (img_dep/data.shape[0], img_rows/data.shape[1], img_cols/data.shape[2])
    data = ndimage.zoom(data, resize_factor, order=0, mode=mode, cval=0.0)
    return data

def image_windowing(img, ww=1800, wl=400):
    """
    preprocessing for CT image (medical)
    Parameters
    - img shape [width, height, depth]
    - ww & wl: bone preset
    """
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
    """
    # inpt shape  : (data)
    # output shape : (data)
    """
    data_nii = np.transpose(data)
    output = nib.Nifti1Image(data_nii, affine=np.eye(4))
    nib.save(output, path)

from skimage.transform import resize
def image_resample_array(src_imgs, src_spacing, target_spacing):

    src_spacing = np.round(src_spacing, 3)
    target_shape = [int(src_imgs.shape[ix] * src_spacing[::-1][ix] / target_spacing[::-1][ix]) for ix in range(len(src_imgs.shape))]
    for i in range(len(target_shape)):
        try:
            assert target_shape[i] > 0
        except:
            raise AssertionError("AssertionError:", src_imgs.shape, src_spacing, target_spacing)

    img = src_imgs.astype(float)
    resampled_img = resize(img, target_shape, order=1, clip=True, mode='edge').astype('float32')

    return resampled_img

def label_binary_dilation(x, radius=3): # 확장
    """ Return fast binary morphological dilation of an image.
    see `skimage.morphology.binary_dilation <http://scikit-image.org/docs/dev/api/skimage.morphology.html#skimage.morphology.binary_dilation>`_.

    Parameters
    -----------
    x : 2D array image.
    radius : int for the radius of mask.
    """
    from skimage.morphology import disk, binary_dilation,dilation
    mask = disk(radius)
    x = dilation(x, selem=mask)
    return x 

# multi-dimensional median filter # 레이블 스무딩
# label = scipy.ndimage.filters.median_filter(label,size=5)

def label_threshold(data,threshold=0.5):
    """
    threholds
    """
    data[data<=threshold]=0
    data[data>threshold] =1
    return data

def label_crop_nonzero_slice(dimension,image,mask,*args):
    if dimension==3:
        xs,ys,zs = np.where(mask!=0)
        image = image[min(xs):max(xs)+1,min(ys):max(ys)+1,min(zs):max(zs)+1]
        mask  = mask[min(xs):max(xs)+1,min(ys):max(ys)+1,min(zs):max(zs)+1]
    elif dimension==4:
        xs,ys,zs,ts = np.where(mask!=0)
        image = image[min(xs):max(xs)+1,min(ys):max(ys)+1,min(zs):max(zs)+1,min(ts):max(ts)+1]
        mask  = mask[min(xs):max(xs)+1,min(ys):max(ys)+1,min(zs):max(zs)+1,min(ts):max(ts)+1]
    else:
        print('dimension should be 3 or 4')
#     print(args[1])
    return image,mask,args[0]

def label_crop_curriculum(image,mask,crop_shape):
    # input shape
    
    image_ = image.copy()
    mask_ = mask.copy()
    mask_[mask_!=0]=0
#     original_shape = image.shape
#     D,H,W = crop_shape
    orig_D, orig_H, orig_W = image.shape
    crop_D, crop_H, crop_W = crop_shape
    
    while np.any(mask_)==False:
        depth_min =int(np.random.rand() * orig_D/2)
        height_min =int(np.random.rand() * orig_H/2)
        width_min = int(np.random.rand() * orig_W/2)
        depth_max = depth_min + crop_D#crop_shape[0]
        height_max =  height_min + crop_H#crop_shape[1]
        width_max = width_min + crop_W#crop_shape[2]
        
        if depth_max > orig_D:
            depth_min -= (depth_max - orig_D)
            depth_max = orig_D
        image_ = image[depth_min:depth_max,height_min:height_max,width_min:width_max]
        mask_ = mask[depth_min:depth_max,height_min:height_max,width_min:width_max]
#         print(height_min,height_max,width_min,width_max)
#     image_ = image_resize(image_,D,H,W,'constant')
#     mask_ = image_resize(label_,D,H,W,'nearest')
    
    return image_,mask_

def label_voxelRemovalExcept1Cluster(inputs):
    # Tensorflow 5D tensor (img_dep,img_cols,img_rows,img_channel)
    # Pytorch 5D tensor (img_channel,img_dep,img_cols,img_rows)
    """
    label_voxel remover for 4D tensor [...,class]
    Remove noise except 1 main voxel    
    """
    outputs = np.zeros((inputs.shape[0],inputs.shape[1],inputs.shape[2],inputs.shape[3]))
    
    for j in range(inputs.shape[-1]):
        outputs[:,:,:,j] = measure.label(inputs[:,:,:,j], background = 0)
        cluster = np.unique(outputs[:,:,:,j])
        cluster_max_voxels = 0

        for k in sorted(cluster)[1:]:
            number_of_voxels = len(outputs[:,:,:,j][outputs[:,:,:,j] == k])
            
            if cluster_max_voxels < number_of_voxels:
                cluster_max_voxels = number_of_voxels
                cluster_max_label = k
                
        outputs[:,:,:,j][outputs[:,:,:,j]!= cluster_max_label] = 0
        outputs[:,:,:,j][outputs[:,:,:,j]!= 0] = 1

    return outputs

def label_onehotEncoding(label,num_class,backend='keras'):
    """
    !!! Must include background(0)
    
    backend='pytorch'
    #input shape  (value  0,1,2,...)   : (image_depth,image_height,image_width)
    #output shape (values 0,1) : (num_class+1,image_depth,image_height,image_width)
    #background is 0, so channel should be num_class+1
    backend='keras'
    #input shape  (value  0,1,2,...)   : (image_depth,image_height,image_width)
    #output shape (values 0,1) : (num_class+1,image_depth,image_height,image_width)
    #background is 0, so channel should be num_class+1
    """

    dimension = len(label.shape)
    if dimension != 3:
        print('error')
    else:
        if backend=='pytorch':
            label_onehot = np.zeros((num_class,label.shape[0],label.shape[1],label.shape[2])) #torch
        else:
            label_onehot = np.zeros((label.shape[0],label.shape[1],label.shape[2],num_class))
               
        for idx in range(num_class):
            label_temp = label.copy() # torch --> clone()
            label_temp[label_temp!=idx]=0.
            label_temp[label_temp!=0.]=1.
            
            if backend=='pytorch':
                label_onehot[idx] = label_temp
            else:
                label_onehot[...,idx] = label_temp
    return label_onehot

# def label_onehot_encode(label,num_class):
    
#     #input shape  (value  0,1,2,...)   : (image_depth,image_height,image_width)
#     #output shape (values 0,1) : (num_class+1,image_depth,image_height,image_width)
#     #background is 0, so channel should be num_class+1
#     num_class += 1
    
#     dimension = len(label.shape)
#     if dimension != 3:
#         print('error')
#     else:
#         label_onehot = torch.zeros((num_class,label.shape[0],label.shape[1],label.shape[2]))
#         for idx in range(num_class):
#             if idx ==0:
#                 #background
#                 label_temp = label.clone()
#                 label_temp[label_temp!=0]=100.
#                 label_temp[label_temp==0]=1
#                 label_temp[label_temp==100.]=0
#                 label_onehot[idx] = label_temp
#             else:
#                 label_temp = label.clone()
#                 label_temp[label_temp!=idx]=0.
#                 label_temp[label_temp!=0.]=1.
#                 label_onehot[idx] = label_temp
#     return label_onehot


def label_onehotDecoding_softmax(label_onehot,backend='keras'):
    """
    !!! Must include background(0)
    
    backend='keras'
    #input shape (values 0,1) : (channel,image_depth,image_height,image_width)
    #output shape (value  0,1,2,...)  : (image_depth,image_height,image_width)
    backend='pytorch'
    #input shape (values 0,1) : (image_depth,image_height,image_width,channel)
    #output shape (value  0,1,2,...)  : (image_depth,image_height,image_width)
    """

    # torch
#     label_onehot = label_onehot.squeeze()
#     value, indices = torch.max(label_onehot,0).astype('float32')
    if backend=='pytorch':
          indices = np.argmax(label_onehot,0).astype('float32') # 0 for channel
    else:
          indices = np.argmax(label_onehot,-1).astype('float32') # 0 for channel
    return indices

def label_onehotDecoding_sigmoid(label_onehot,backend='keras'):
    """
    !!! Must not include background(0)
    """
    
    if backend=='pytorch':  
        label = np.zeros((label_onehot.shape[1],label_onehot.shape[2],label_onehot.shape[3]))
        for idx in range(label_onehot.shape[0]):
            label_temp = label_onehot[idx]
            label_temp[label_temp!=1.]=0.
            label_temp[label_temp==1.]=idx+1
            label += label_temp
    else:
        label = np.zeros((label_onehot.shape[0],label_onehot.shape[1],label_onehot.shape[2]))
        for idx in range(label_onehot.shape[-1]):
            label_temp = label_onehot[...,idx]
            label_temp[label_temp!=1.]=0.
            label_temp[label_temp==1.]=idx+1
            label += label_temp
            
            if len(label[label>idx+1])>0
                label_1 = label_getCentroid()
                label_2 = label_getCentroid()
            
    return label

def label_getCentroid(label):
    """Input Label should be binary and numpy array"""
    
    count= len(np.argwhere(label == 1))
    if count ==0:
        print('Label does not have the value 1')
        
    if len(label.shape)==2:
        center_height, center_width = np.argwhere(label == 1).sum(0)/count
        return center_height, center_width
    elif len(label.shape)==3:
        center_depth,center_height, center_width = np.argwhere(label == 1).sum(0)/count
        return center_depth,center_height, center_width
    
def label_RemoveNonLabeledSlice(image,label,reference_label):
    """
    input : squential image & label (depth,height,width,channel) tensorflow
    TODO --> Multi reference label
    """
        
    for idx_depth in reversed(range(len(label))):
        if not np.any(label[idx_depth,:,:,reference_label]): # TODO multiple channel
            image = np.delete(image,idx_depth,axis=0)
            label = np.delete(label,idx_depth,axis=0)

    return image, label

def dataset_buildNearByStack_3ch(image):
    # shape (batch,height,width,1)
    # last axis == channel
    image_top = image.copy()
    image_top = np.insert(image_top,0,0,axis=0)
    image_top = np.delete(image_top,-1,axis=0)
    
    image_middle = image.copy()
    
    image_bottom = image.copy()
    image_bottom = np.insert(image_bottom,-1,0,axis=0)
    image_bottom = np.delete(image_bottom,0,axis=0)
    
    image = np.concatenate((image_top,image_middle,image_bottom),axis=-1)
    return image

