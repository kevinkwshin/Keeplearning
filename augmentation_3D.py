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
