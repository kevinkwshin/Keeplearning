def image_windowing(img, ww=1800, wl=400):
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
