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
    rotation_matrix_y = np.array([[1, 0, 0, 0],
                                              [0, 1, 0, 0],
                                              [0, 0, 1, 0],
                                              [0, 0, 0, 1]])
    rotation_matrix_x = np.array([[1, 0, 0, 0],
    	                          [0, np.cos(theta3), -np.sin(theta3), 0],
                                  [0, np.sin(theta3), np.cos(theta3), 0],
                                  [0, 0, 0, 1]])
    rotation_matrix = np.dot(np.dot(rotation_matrix_y, rotation_matrix_z), rotation_matrix_x)
    print(x.shape)

    h, w, d = x.shape[row_index], x.shape[col_index], x.shape[dep_index]
    transform_matrix = transform_matrix_offset_center(rotation_matrix, d, w, h)
    x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval)
    return x


def transform_matrix_offset_center(matrix, x, y, z):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    o_z = float(z) / 2 + 0.5
    offset_matrix = np.array([[1, 0, 0, o_x], [0, 1, 0, o_y], [0, 0, 1, o_z], [0, 0, 0, 1]])
    reset_matrix = np.array([[1, 0, 0, -o_x], [0, 1, 0, -o_y], [0, 0, 1, -o_z], [0, 0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def apply_transform(x, transform_matrix, channel_index=0, fill_mode='nearest', cval=0.):
    x = np.rollaxis(x, channel_index, 0)
    final_affine_matrix = transform_matrix[:3, :3]
    final_offset = transform_matrix[:3, 3]
    channel_images = [scipy.ndimage.interpolation.affine_transform(x_channel, final_affine_matrix,
                      final_offset, order=0, mode=fill_mode, cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_index+1)
    return x
  
  
