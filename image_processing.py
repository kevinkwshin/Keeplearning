import torch
import numpy as np
import SimpleITK as sitk
import nibabel as nib
from scipy import ndimage
from skimage import measure
from skimage.measure import label, regionprops
import matplotlib.patches as mpatches

def z_normalization(img, num_channels):
    """
    z_normalization test
    """
    for i in range(num_channels):
        img[..., i] -= np.mean(img[..., i])
        img[..., i] /= np.std(img[..., i])
    return img

def sample_z_norm(data, mean, sd):
    data -= mean
    data /= sd
    return data

def image_CT_preprocess_float(x,x_cutoff_max=100,x_cutoff_min=0):
    """
    Scale image to range 0..1 for correct plot
    """
    
    x[x < -1024] = -1024.
    x[x >= 3071] = 3071.
    
    x_max = np.percentile(x, x_cutoff_max)
    x_min = np.percentile(x, x_cutoff_min)    
    if x_max - x_min != 0:
        x = (x - x_min) / (x_max - x_min)
        x = x.clip(0, 1).astype('float32')
    elif x_max - x_min == 0:
        x = np.zeros_like(x)
    return x

def image_preprocess_uint8(x,x_cutoff_max=100,x_cutoff_min=0):
    """
    Cut off & Convert image array into 0~255
    """
    x_max = np.percentile(x, x_cutoff_max)
    x_min = np.percentile(x, x_cutoff_min)
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    x= np.round(x*255)
    return x

def image_preprocess_CT_uint8(img):
    """
    Convert CT image array in HU into 0~255 uint
    """
    if np.min(img)<0:
        img[img < -1024] = -1024.
        img[img >= 3071] = 3071.
        img += 1024.
    img /= (2**12-1)
    return img

def image_resize3D(data, img_dep=200., img_rows=200., img_cols=200.): # 3D image
    """
    Resizer for 3D image
    """
    resize_factor = (img_dep/data.shape[0], img_rows/data.shape[1], img_cols/data.shape[2])
    data = ndimage.zoom(data, resize_factor, order=0, mode='constant', cval=0.0)
    return data

def label_resize3D(data, img_dep, img_rows, img_cols, median_filter=False): # 3D image
    """
    Resizer for 3D image
    median_filter --> when voxel spaing is greater x 5 
    """
    resize_factor = (img_dep/data.shape[0], img_rows/data.shape[1], img_cols/data.shape[2])
    data = ndimage.zoom(data, resize_factor, order=0, mode='nearest', cval=0.0)
    if median_filter==True:
        data = ndimage.filters.median_filter(data,size=(1,5,5), mode='constant')
    return data

def image_windowing_CT(img, ww=1800, wl=400):
    """
    preprocessing for CT image (medical)
    Parameters
    - img shape [width, height, depth] <- 3D
    - ww & wl: bone preset
    """
    
    maxp = np.max(img)
    minp = np.min(img)
    
    if minp >=0:
        img=img - 1024
    else:
        img[img < -1024] = -1024.
        img[img >= 3071] = 3071.

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
    A quick array saving function into nii or nii.gz
    - input shape  : (data)
    - output shape : (data)
    """
    data_nii = np.transpose(data)
    output = nib.Nifti1Image(data_nii, affine=np.eye(4))
    nib.save(output, path)

from skimage.transform import resize
def image_resample_array(src_imgs, src_spacing, target_spacing):

    src_spacing = np.round(src_spacing, 3)
    target_shape = [int(src_imgs.shape[ix] * src_spacing[::-1][ix] / target_spacing[::-1][ix]) 
                    for ix in range(len(src_imgs.shape))]
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
    see `skimage.morphology.binary_dilation'

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
    data = data.copy()
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
    """
    original_shape = image.shape
    D,H,W = crop_shape
    """
    image_ = image.copy()
    mask_ = mask.copy()
    mask_[mask_!=0]=0
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

def label_voxelRemovalExcept1Cluster(inputs,backend='keras'):
    """
    TODO : pytorch version is not avaiable!!
    label_voxel remover for 4D tensor [...,class] for keras
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


def label_onehotDecodingWithBackground(label_onehot,backend='keras'):
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

def label_onehotDecodingWithoutBackground(label_onehot,backend='keras'):
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
            label[label>idx+1] = idx
            
    return label

def label_coordinateGetCentroid(label):
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
    
    
def label_coordinate3DMinMax(inputs):
    """
    return : (min), (max)
    """
    coord1_min,coord2_min,coord3_min = np.unravel_index(np.argmin(inputs),inputs.shape)
    coord1_max,coord2_max,coord3_max = np.unravel_index(np.argmax(inputs),inputs.shape)
    return np.array([coord1_min,coord2_min,coord3_min,coord1_max,coord2_max,coord3_max])
    
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


def label_thresholdEachChannel(data,threshold_array,backend='keras'):
    # input shape : (...,channel)
    # output shape : (...,channel)
    if backend == 'keras':
        if len(data.shape[-1])!=len(threshold_array):
            print('not proper input array')

        for idx_channel in range(data.shape[-1]):

            data_temp = data[...,idx_channel].copy()
            data_temp[data_temp< threshold_array[idx_channel]] = 0
            data_temp[data_temp>=threshold_array[idx_channel]] = 1

            data[...,idx_channel] = data_temp
    else:
        print('pytorch version not available')
        
    return data

def label_getROIs(mask, ignoring_pixel=200, plot_image=False):
    """
    return [(min_height, min_width, max_height, max_width)]
    """
    mask = label_threshold(mask)
    mask_labeled, mask_count = label(mask,return_num=True)
    
    if plot_image:
        fig, ax1 = plt.subplots()
    ROIs = []
    for region in regionprops(mask_labeled):
        if region.area >= ignoring_pixel:
            min_height, min_width, max_height, max_width = region.bbox
            rect = mpatches.Rectangle((min_width, min_height), max_width - min_width, max_height - min_height,
                                      fill=False, edgecolor='red', linewidth=1)
            ROIs.append((min_height, min_width, max_height, max_width))

            if plot_image:
                ax1.imshow(mask)
                ax1.add_patch(rect)
    return ROIs
