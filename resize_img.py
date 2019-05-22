def resize(data, img_dep=200., img_rows=200., img_cols=200.):
    resize_factor = (img_dep/data.shape[0], img_rows/data.shape[1], img_cols/data.shape[2])
    data = ndimage.zoom(data, resize_factor, order=0, mode='constant', cval=0.0)
    return data
