import numpy as np
import numpy as np
import re
from scipy import linalg, ndimage
import scipy.ndimage
from six.moves import range
import os
import threading
import SimpleITK as sitk


# if backend == 'pytorch'
#     self.channel_index = 1
#     self.dep_index = 2
#     self.row_index = 3
#     self.col_index = 4
# if backend == 'keras'
#     self.channel_index = 4
#     self.dep_index = 1
#     self.row_index = 2
#     self.col_index = 3

# def random_rotation_3D(x, rg, row_index=2, col_index=3, dep_index = 1, channel_index=0,
#                     fill_mode='nearest', cval=0.):
  
def random_rotation_3D(x, rg, backend='keras', fill_mode='nearest', cval=0.):
    """
    input : 5D tensor
    rg : (float,float,float)
    
    """
    if backend=='keras':
         dep_index=1
         row_index=2
         col_index=3
         channel_index=4
    else:
         dep_index=2
         row_index=3
         col_index=4
         channel_index=1
    
    theta1 = np.pi / 180 * np.random.uniform(-rg[0], rg[0])
    theta2 = np.pi / 180 * np.random.uniform(-rg[1], rg[1])
    theta3 = np.pi / 180 * np.random.uniform(-rg[2], rg[2])
    print(theta1,theta2,theta3)


    rotation_matrix_z = np.array([[np.cos(theta1), -np.sin(theta1), 0, 0],
                                  [np.sin(theta1), np.cos(theta1), 0, 0],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1]])
    rotation_matrix_y = np.array([[np.cos(theta2), 0, -np.sin(theta2), 0],
                                              [0, 1, 0, 0],
                                              [np.sin(theta2), 0, np.cos(theta2), 0],
                                              [0, 0, 0, 1]])

def random_rotation(x, rg, row_index=2, col_index=3, dep_index = 1, channel_index=0,
                    fill_mode='nearest', cval=0.):
    theta1 = np.pi / 180 * np.random.uniform(-rg[0], rg[0])
    theta2 = np.pi / 180 * np.random.uniform(-rg[1], rg[1])
    theta3 = np.pi / 180 * np.random.uniform(-rg[2], rg[2])

    rotation_matrix_z = np.array([[np.cos(theta1), -np.sin(theta1), 0, 0],
                                  [np.sin(theta1), np.cos(theta1), 0, 0],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1]])
    rotation_matrix_y = np.array([[np.cos(theta2), 0, -np.sin(theta2), 0],
                                              [0, 1, 0, 0],
                                              [np.sin(theta2), 0, np.cos(theta2), 0],
                                              [0, 0, 0, 1]])
    rotation_matrix_y = np.array([[1, 0, 0, 0],
                                              [0, 1, 0, 0],
                                              [0, 0, 1, 0],
                                              [0, 0, 0, 1]])
    rotation_matrix_x = np.array([[1, 0, 0, 0],
    	                          [0, np.cos(theta3), -np.sin(theta3), 0],
                                  [0, np.sin(theta3), np.cos(theta3), 0],
                                  [0, 0, 0, 1]])
    rotation_matrix = np.dot(np.dot(rotation_matrix_y, rotation_matrix_z), rotation_matrix_x)

    h, w, d = x.shape[row_index], x.shape[col_index], x.shape[dep_index]
    transform_matrix = transform_matrix_offset_center(rotation_matrix, d, w, h)
    x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval)
    return x


def random_shift(x, wrg, hrg, drg, row_index=2, col_index=3, dep_index = 1, channel_index=0,
                 fill_mode='nearest', cval=0.):
    h, w, d = x.shape[row_index], x.shape[col_index], x.shape[dep_index]
    tx = np.random.uniform(-hrg, hrg) * h
    ty = np.random.uniform(-wrg, wrg) * w
    tz = np.random.uniform(-drg, drg) * d
    translation_matrix = np.array([[1, 0, 0, tz],
                                   [0, 1, 0, ty],
                                   [0, 0, 1, tx],
                                   [0, 0, 0, 1]])

    transform_matrix = translation_matrix  # no need to do offset
    x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval)
    return x


def random_shear(x, intensity, row_index=2, col_index=3, dep_index=1, channel_index=0,
                 fill_mode='nearest', cval=0.):
    shear1 = np.random.uniform(-intensity, intensity)
    shear2 = np.random.uniform(-intensity, intensity)
    shear3 = np.random.uniform(-intensity, intensity)
    shear_matrix_z = np.array([[np.cos(shear1),0, 0, 0],
                             [-np.sin(shear1), 1, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]])
    shear_matrix_x = np.array([[1,-np.sin(shear2), 0, 0],
                             [0, np.cos(shear2), 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]])
    shear_matrix_y = np.array([[1,0, -np.sin(shear3), 0],
                             [0, 1, 0, 0],
                             [0, 0, np.cos(shear3), 0],
                             [0, 0, 0, 1]])
    shear_matrix = np.dot(np.dot(shear_matrix_y, shear_matrix_z), shear_matrix_x)

    h, w, d = x.shape[row_index], x.shape[col_index], x.shape[dep_index]
    transform_matrix = transform_matrix_offset_center(shear_matrix, d, w, h)
    x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval)
    return x


def random_zoom(x, zoom_range, row_index=2, col_index=3, dep_index = 1, channel_index=0,
                fill_mode='nearest', cval=0.):
    if len(zoom_range) != 2:
        raise Exception('zoom_range should be a tuple or list of two floats. '
                        'Received arg: ', zoom_range)

    if zoom_range[0] == 1 and zoom_range[1] == 1:
        zx, zy, zz = 1, 1, 1
    else:
        zx, zy, zz = np.random.uniform(zoom_range[0], zoom_range[1], 3)
    zoom_matrix = np.array([[zz, 0, 0, 0],
                            [0, zy, 0, 0],
                            [0, 0, zx, 0],
                            [0, 0, 0, 1]])

    h, w, d = x.shape[row_index], x.shape[col_index], x.shape[dep_index]
    transform_matrix = transform_matrix_offset_center(zoom_matrix, d, w, h)
    x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval)
    return x


def random_barrel_transform(x, intensity):
    # TODO
    pass


def random_channel_shift(x, intensity, channel_index=0):
    x = np.rollaxis(x, channel_index, 0)
    min_x, max_x = np.min(x), np.max(x)
    channel_images = [np.clip(x_channel + np.random.uniform(-intensity, intensity), min_x, max_x)
                      for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_index+1)
    return x


def transform_matrix_offset_center(matrix, x, y, z):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    o_z = float(z) / 2 + 0.5
    offset_matrix = np.array([[1, 0, 0, o_x], [0, 1, 0, o_y], [0, 0, 1, o_z], [0, 0, 0, 1]])
    reset_matrix = np.array([[1, 0, 0, -o_x], [0, 1, 0, -o_y], [0, 0, 1, -o_z], [0, 0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def apply_transform(x, transform_matrix, channel_index=4, fill_mode='nearest', cval=0.):
    x = np.rollaxis(x, channel_index, 0)
    final_affine_matrix = transform_matrix[:3, :3]
    final_offset = transform_matrix[:3, 3]
    channel_images = [scipy.ndimage.interpolation.affine_transform(x_channel, final_affine_matrix,
                      final_offset, order=0, mode=fill_mode, cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_index+1)
    return x
  
  

  
class augaug():
    '''Generate minibatches with
    real-time data augmentation.
    # Arguments
        featurewise_center: set input mean to 0 over the dataset.
        samplewise_center: set each sample mean to 0.
        featurewise_std_normalization: divide inputs by std of the dataset.
        samplewise_std_normalization: divide each input by its std.
        zca_whitening: apply ZCA whitening.
        rotation_range: degrees (0 to 180).
        width_shift_range: fraction of total width.
        height_shift_range: fraction of total height.
        shear_range: shear intensity (shear angle in radians).
        zoom_range: amount of zoom. if scalar z, zoom will be randomly picked
            in the range [1-z, 1+z]. A sequence of two can be passed instead
            to select this range.
        channel_shift_range: shift range for each channels.
        fill_mode: points outside the boundaries are filled according to the
            given mode ('constant', 'nearest', 'reflect' or 'wrap'). Default
            is 'nearest'.
        cval: value used for points outside the boundaries when fill_mode is
            'constant'. Default is 0.
        horizontal_flip: whether to randomly flip images horizontally.
        vertical_flip: whether to randomly flip images vertically.
        rescale: rescaling factor. If None or 0, no rescaling is applied,
            otherwise we multiply the data by the value provided (before applying
            any other transformation).
        dim_ordering: 'th' or 'tf'. In 'th' mode, the channels dimension
            (the depth) is at index 1, in 'tf' mode it is at index 3.
            It defaults to the `image_dim_ordering` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "th".
    '''
    def __init__(self,
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False,
                 rotation_range=0.,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 depth_shift_range=0.,
                 shear_range=0.,
                 zoom_range=0.,
                 channel_shift_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 depthly_flip=False,
                 rescale=None,
                 backend='keras):
   
      if backend=='keras':
           dep_index=1
           row_index=2
           col_index=3
           channel_index=4
      else:
           dep_index=2
           row_index=3
           col_index=4
           channel_index=1
                 
    def random_transform(self, x):
        # x is a single image, so it doesn't have image number at index 0
        img_dep_index = self.dep_index - 1
        img_row_index = self.row_index - 1
        img_col_index = self.col_index - 1
        img_channel_index = self.channel_index - 1

        transform_matrix = None
        # use composition of homographies to generate final transform that needs to be applied
        if self.rotation_range:
            theta1 = np.pi / 180 * np.random.uniform(-self.rotation_range[0], self.rotation_range[0])
            theta2 = np.pi / 180 * np.random.uniform(-self.rotation_range[1], self.rotation_range[1])
            theta3 = np.pi / 180 * np.random.uniform(-self.rotation_range[2], self.rotation_range[2])

            rotation_matrix_z = np.array([[np.cos(theta1), -np.sin(theta1), 0, 0],
                                      [np.sin(theta1), np.cos(theta1), 0, 0],
                                      [0, 0, 1, 0],
                                      [0, 0, 0, 1]])
            rotation_matrix_y = np.array([[np.cos(theta2), 0, -np.sin(theta2), 0],
                                          [0, 1, 0, 0],
                                          [np.sin(theta2), 0, np.cos(theta2), 0],
                                          [0, 0, 0, 1]])
            rotation_matrix_x = np.array([[1, 0, 0, 0],
                                          [0, np.cos(theta3), -np.sin(theta3), 0],
                                          [0, np.sin(theta3), np.cos(theta3), 0],
                                          [0, 0, 0, 1]])
            rotation_matrix = np.dot(np.dot(rotation_matrix_y, rotation_matrix_z), rotation_matrix_x)

            transform_matrix = rotation_matrix if transform_matrix is None else np.dot(transform_matrix, rotation_matrix)
        
        if self.height_shift_range:
            tx = np.random.uniform(-self.height_shift_range, self.height_shift_range) * x.shape[img_row_index]
        else:
            tx = 0
        if self.width_shift_range:
            ty = np.random.uniform(-self.width_shift_range, self.width_shift_range) * x.shape[img_col_index]
        else:
            ty = 0
        if self.depth_shift_range:
            tz = np.random.uniform(-self.depth_shift_range, self.depth_shift_range) * x.shape[img_dep_index]
        else:
            tz = 0

        if self.shear_range:
            shear1 = np.random.uniform(-self.shear_range, self.shear_range)
            shear2 = np.random.uniform(-self.shear_range, self.shear_range)
            shear3 = np.random.uniform(-self.shear_range, self.shear_range)
            shear_matrix_z = np.array([[np.cos(shear1),0, 0, 0],
                             [-np.sin(shear1), 1, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]])
            shear_matrix_x = np.array([[1,-np.sin(shear2), 0, 0],
                                     [0, np.cos(shear2), 0, 0],
                                     [0, 0, 1, 0],
                                     [0, 0, 0, 1]])
            shear_matrix_y = np.array([[1,0, -np.sin(shear3), 0],
                                     [0, 1, 0, 0],
                                     [0, 0, np.cos(shear3), 0],
                                     [0, 0, 0, 1]])
            shear_matrix = np.dot(np.dot(shear_matrix_y, shear_matrix_z), shear_matrix_x)
            transform_matrix = shear_matrix if transform_matrix is None else np.dot(transform_matrix, shear_matrix)

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy, zz = 1, 1, 1
        else:
            zx, zy, zz = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 3)

        if tx != 0 or ty != 0 or tz != 0:
            shift_matrix = np.array([[1, 0, 0, tz],
                                     [0, 1, 0, ty],
                                     [0, 0, 1, tx],
                                     [0, 0, 0, 1]])
            transform_matrix = shift_matrix if transform_matrix is None else np.dot(transform_matrix, shift_matrix)

        if zx != 1 or zy != 1 or zz != 1:
            zoom_matrix = np.array([[zz, 0, 0, 0],
                                    [0, zy, 0, 0],
                                    [0, 0, zx, 0],
                                    [0, 0, 0, 1]])
            transform_matrix = zoom_matrix if transform_matrix is None else np.dot(transform_matrix, zoom_matrix)

        h, w, d = x.shape[img_row_index], x.shape[img_col_index], x.shape[img_dep_index]
        transform_matrix = transform_matrix_offset_center(transform_matrix, d, w, h)
        x = apply_transform(x, transform_matrix, img_channel_index,
                            fill_mode=self.fill_mode, cval=self.cval)
        if self.channel_shift_range != 0:
            x = random_channel_shift(x, self.channel_shift_range, img_channel_index)

        if self.horizontal_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_col_index)

        if self.vertical_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_row_index)

        if self.depthly_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_dep_index)

        # TODO:
        # channel-wise normalization
        # barrel/fisheye
        return x
