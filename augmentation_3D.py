import numpy as np
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
  
def random_rotation_3D(x, rg, backend='keras',
                    fill_mode='nearest', cval=0.):
    """
    rg : (float,float,float)
    
    """
    dep_index=1,
    row_index=2,
    col_index=3, 
    channel_index=0,
    
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
  
  
