# 3D Augmentation using SimpleITK

import SimpleITK as sitk
import numpy as np

#############################################################################################################################
# BSpline Transform
def augmentation_bspline_tranform_parameter(sitk_image, MeshSize=6, scale_distortion=1):
    
    transformDomainMeshSize=[MeshSize]*3
    tx = sitk.BSplineTransformInitializer(sitk_image, transformDomainMeshSize)
    params = tx.GetParameters()
    paramsNp=np.asarray(params,dtype=float)
    paramsNp = paramsNp + np.random.randn(paramsNp.shape[0])*scale_distortion
#     paramsNp[0:int(len(params)/3)]=0 #remove z deformations! The resolution in z is too bad
    params=tuple(paramsNp)
    tx.SetParameters(params)

    return tx

def augmentation_bspline_tranform(sitk_input,tx,interpolator=sitk.sitkBSpline):
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(sitk_input)
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(tx)
    resampler.SetDefaultPixelValue(0)
    
    aug_input = resampler.Execute(sitk_input)
    
    return aug_input

# Usage
# image = sitk.ReadImage(path_image)
# tx = bspline_tranform_parameter(image)
# aug_image = bspline_tranform(image,tx,sitk.sitkBSpline)
# aug_image = sitk.GetArrayFromImage(aug_image)
# aug_mask = bspline_tranform(mask,tx,sitk.sitkNearestNeighbor)
# aug_mask = sitk.GetArrayFromImage(aug_mask)

#############################################################################################################################
# Affine Transform
def augmentation_affine_transform_parameter(translation_3d,size_scale,matrix_scale):
    affine_center = (0,0,0)
    affine_translation = (translation_3d[0]*np.random.random(),
                          translation_3d[1]*np.random.random(),
                          translation_3d[2]*np.random.random())
    affine_scale = 1 + size_scale*np.random.random()
    affine_matrix = [1,0,0,0,1,0,0,0,1] - np.random.random(9)*matrix_scale
    return affine_matrix,affine_translation,affine_center,affine_scale

def augmentation_affine_transform(sitk_input,interpolator):
    transform = sitk.AffineTransform(affine_matrix, affine_translation, affine_center)
    transform.Scale(affine_scale)

    reference_image = image
    aug_input = sitk.Resample(sitk_input, reference_image, transform, interpolator)
    
    return aug_input


# Usage
# affine_matrix,affine_translation,affine_center,affine_scale= affine_transform_parameter((0,10,10),0.05,0.05)
# aug_image = affine_transform(image,sitk.sitkBSpline)
# aug_mask  = affine_transform(mask,sitk.sitkNearestNeighbor)
