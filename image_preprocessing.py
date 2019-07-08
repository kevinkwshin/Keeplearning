def _normalize(self,slice):
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

def preprocess_img(img):
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

def preprocess_CT_256(img):
    img[img < -1024] = -1024.
    img[img >= 3071] = 3071.
    img += 1024.
    img /= (2**12-1)
    return img
