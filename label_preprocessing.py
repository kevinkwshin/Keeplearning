def binary_dilation(x, radius=3): # 확장
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
label = scipy.ndimage.filters.median_filter(label,size=5)
