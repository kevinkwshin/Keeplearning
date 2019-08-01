# 3D Augmentation using SimpleITK

def bspline_tranform_parameter(sitk_image, MeshSize=6, scale_distortion=4):
    
    transformDomainMeshSize=[MeshSize]*3
    tx = sitk.BSplineTransformInitializer(sitk_image, transformDomainMeshSize)
    params = tx.GetParameters()
    paramsNp=np.asarray(params,dtype=float)
    paramsNp = paramsNp + np.random.randn(paramsNp.shape[0])*scale_distortion
#     paramsNp[0:int(len(params)/3)]=0 #remove z deformations! The resolution in z is too bad
    params=tuple(paramsNp)
    tx.SetParameters(params)

    return tx

def bspline_tranform(sitk_input,tx,interpolator):
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(sitk_input)
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(tx)
    resampler.SetDefaultPixelValue(0)
    
    aug_input = resampler.Execute(sitk_input)
    
    return aug_input
