import torch
import numpy as np

def label_binary_dilation(x, radius=3): # 확장
    """ Return fast binary morphological dilation of an image.
    see `skimage.morphology.binary_dilation <http://scikit-image.org/docs/dev/api/skimage.morphology.html#skimage.morphology.binary_dilation>`_.

    Parameters
    -----------
    x : 2D array image.
    radius : int for the radius of mask.
    """
    from skimage.morphology import disk, binary_dilation
    mask = disk(radius)
    x = binary_dilation(x, selem=mask)
    return x 

# multi-dimensional median filter # 레이블 스무딩
# label = scipy.ndimage.filters.median_filter(label,size=5)

def label_threshold(data,threshold):
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
    original_shape = image.shape
    
    while np.any(mask_)==False:
        depth_min =int(np.random.rand() * original_shape[0]/2)
        height_min =int(np.random.rand() * original_shape[1]/2)
        width_min = int(np.random.rand() * original_shape[2]/2)
        depth_max = depth_min + crop_shape[0]
        height_max =  height_min + crop_shape[1]
        width_max = width_min + crop_shape[2]
        image_ = image[depth_min:depth_max,height_min:height_max,width_min:width_max]
        mask_ = mask[depth_min:depth_max,height_min:height_max,width_min:width_max]
#         print(height_min,height_max,width_min,width_max)
    return image_,mask_


def label_voxel_remover(results):
    # Tensorflow 5D tensor (batch,img_dep,img_cols,img_rows,img_channel)
    # Pytorch 5D tensor (batch,img_channel,img_dep,img_cols,img_rows)
    results_processed = np.zeros((results.shape[0],results.shape[1],results.shape[2],results.shape[3],results.shape[4]))
    
    for i in range(results.shape[0]):
        for j in range(results.shape[-1]):
            results_processed[i,:,:,:,j] = measure.label(results[i,:,:,:,j], background = 0)
            cluster = np.unique(results_processed[i,:,:,:,j])
            cluster_max_voxels = 0

            for k in sorted(cluster)[1:]:
                number_of_voxels = len(results_processed[i,:,:,:,j][results_processed[i,:,:,:,j] == k])
                if cluster_max_voxels < number_of_voxels:
                    cluster_max_voxels = number_of_voxels
                    cluster_max_label = k
                    
            results_processed[i,:,:,:,j][results_processed[i,:,:,:,j]!= cluster_max_label] = 0
            results_processed[i,:,:,:,j][results_processed[i,:,:,:,j]!= 0] = 1 # added
    return results_processed

def label_onehot_encode(label,num_class):
    
    #input shape  (value  0,1,2,...)   : (image_depth,image_height,image_width)
    #output shape (values 0,1) : (channel,image_depth,image_height,image_width)
    
    dimension = len(label.shape)
    if dimension != 3:
        print('error')
    else:
        label_onehot = torch.zeros((num_class,label.shape[0],label.shape[1],label.shape[2]))
        for idx in range(num_class):
            label_ = label.clone()
            label_temp = label_
            label_temp[label_temp!=idx+1.]=0.
            label_temp[label_temp!=0.]=1.
            label_onehot[idx] = label_temp
    return label_onehot

# def label_onehot_decode(label_onehot):
#     #input shape (values 0,1) : (channel,image_depth,image_height,image_width)
#     #output shape (value  0,1,2,...)  : (image_depth,image_height,image_width)

#     # torch
# #     label_onehot = label_onehot.squeeze()
# #     value, indices = torch.max(label_onehot,0).astype('float32')

#     # numpy
#     indices = np.argmax(label_onehot,0).astype('float32') # 0 for channel

#     return indices

def label_onehot_decode(label_onehot):
    #input shape (values 0,1) : (channel,image_depth,image_height,image_width)
    #output shape (value  0,1,2,...)  : (image_depth,image_height,image_width)

    # torch
#     label_onehot = label_onehot.squeeze()
#     value, indices = torch.max(label_onehot,0).astype('float32')

    # numpy
    
    label = numpy.zeros((label_onehot.shape[1],label_onehot.shape[2],label_onehot.shape[3]))
    for idx in range(len(label_onehot)):
        label_temp = label_onehot[idx]
        label_temp[label_temp!=idx+1.]=0.
        label_temp[label_temp==idx+1.]=1.
        label += label_temp
#     label = np.argmax(label_onehot,0).astype('float32') # 0 for channel

    return indices
