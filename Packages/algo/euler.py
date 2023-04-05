from lazy_imports import np
import math
# from util.tensors import tens_interp, tens_interp_3d, vect_interp_2d, vect_interp_3d, eigen_vec, eigen_vec_3d
# from util.tensors import direction, direction_3d
from util.tensors import *
from data import io
from util import diff, riemann


def get_eigenvec_at_point(x, y, tensor_field, prev_angle):
  # return first and second component of eigenvector at a point, and associated angle
  tens = tens_interp(x, y, tensor_field)

  # important!!!!!!!!!!!!
  '''
  When using the eigenvector calculate by myself:
  Because when the principal eigenvector is almost horizontal, say at the top of the annulus,
  the eigenvector becomes extremely small, like [0.009986493070520448 1.9950060037743356e-05]
  so we have to adjust it to [1 0] manually.
  When the tensor is very vertical or horizontal, it's typically [6 0; 0 1] or [1 0; 0 6]
  '''
  # if abs(tens[0,1]) < 0.01 and abs(tens[1,0]) < 0.01:
  #   print("small b and c")
  #   u, v = eigen_vec(tens)
  #   print(u, v)
  #   # print(x, y)
  #   u = 1
  #   v = 0
  # else:
  #   u, v = eigen_vec(tens)
  u, v = eigen_vec(tens)

  # important too!!!!!!!!
  angle1 = math.atan2(v, u)
  angle2 = math.atan2(-v, -u)
  if abs(angle1 - prev_angle) < abs(angle2 - prev_angle):
    # keep the sign of eigenvector
    new_angle = angle1
  else:
    u = -u
    v = -v
    new_angle = angle2
  return(u, v, new_angle)

def get_vec_at_point_2d(x, y, vector_field, prev_angle):
  # return first and second component of eigenvector at a point, and associated angle
  vect = vect_interp_2d(x, y, vector_field)

  # important!!!!!!!!!!!!
  '''
  When using the eigenvector calculate by myself:
  Because when the principal eigenvector is almost horizontal, say at the top of the annulus,
  the eigenvector becomes extremely small, like [0.009986493070520448 1.9950060037743356e-05]
  so we have to adjust it to [1 0] manually.
  When the tensor is very vertical or horizontal, it's typically [6 0; 0 1] or [1 0; 0 6]
  '''
  u, v = vect[0], vect[1]

  # important too!!!!!!!!
  angle1 = math.atan2(v, u)
  angle2 = math.atan2(-v, -u)
# 2022.07.30 revised
  new_angle = angle1
#   if abs(angle1 - prev_angle) < abs(angle2 - prev_angle):
#     # keep the sign of eigenvector
#     new_angle = angle1
#   else:
#     u = -u
#     v = -v
#     new_angle = angle2
  return(u, v, new_angle)

def eulerpath_vectbase_2d(vector_field, mask_image, start_coordinate, delta_t=0.25, iter_num=700, filename = '', both_directions=False):
  # calculating first eigenvector
  (x, y) = start_coordinate  
  (u, v) = vect_interp_2d(x, y, vector_field)

  if both_directions:
    back_x, back_y = eulerpath_vectbase_2d(vector_field, mask_image, start_coordinate, -delta_t, iter_num, filename, both_directions=False)
  print("Euler starting eigenvector:", [u,v])
  prev_angle = math.atan2(v, u)

  points_x = []
  points_y = []
    
  gamma = np.zeros((iter_num,2))
  gamma_dot = np.zeros((iter_num,2))
    
  for i in range(iter_num):
    '''
    The reason why x should -v*delta_t instead of +v*delta_t is that: in calculation, we regard upper left
    namely the cell[0,0] as the origin. However, the vector field derived by tensor field regards down left 
    as the origin, namely the cell[size[0]-1,0], only by this can the the value in vector field make sense.
    '''
    
    uk1 = u
    vk1 = v
    x = x + (uk1) * delta_t
    y = y + (vk1) * delta_t
    
    gamma[i,0] = x
    gamma[i,1] = y
    gamma_dot[i,0] = uk1
    gamma_dot[i,1] = vk1
    
    if (math.ceil(x) >= 0 and math.ceil(x) < vector_field.shape[1]
        and math.ceil(y) >= 0 and math.ceil(y) < vector_field.shape[2]
        and mask_image[int(math.ceil(x)), int(math.ceil(y))] > 0):
      points_x.append(x)
      points_y.append(y)
    else:
      break

    (u, v, prev_angle) = get_vec_at_point_2d(x, y, vector_field, prev_angle)

  if both_directions:
#     points_x = points_x[::-1] + back_x
#     points_y = points_y[::-1] + back_y
    points_x = np.concatenate((points_x[::-1], back_x),0)
    points_y = np.concatenate((points_y[::-1], back_y),0)
    
  if filename:
    io.writePath2D(points_x, points_y, filename)

  return np.array(points_x), np.array(points_y)

def eulerpath_vectbase_2d_w_dv(vector_lin, mask_image, start_coordinate, delta_t=0.25, iter_num=700, filename = '', both_directions=False):
  if both_directions:
    back_x, back_y = eulerpath_vectbase_2d_w_dv(-vector_lin, mask_image, start_coordinate, delta_t, iter_num, filename, both_directions=False)

  points_x = []
  points_y = []
  gamma = np.zeros((iter_num,2))
  gamma_dot = np.zeros((iter_num,2))
  gamma[0] = start_coordinate
  gamma_dot[0] = vect_interp_2d(gamma[0,0], gamma[0,1], vector_lin)
  prev_angle = math.atan2(gamma_dot[0,1], gamma_dot[0,0])
  print("Euler starting eigenvector:", gamma_dot[0])
  
  DV = riemann.get_jacobian_2d(vector_lin, mask_image)
  dvv = np.einsum('...ij,j...->i...', DV, vector_lin)
  # calculating following eigenvectors
  for i in range(1,iter_num):
    '''
    The reason why x should -v*delta_t instead of +v*delta_t is that: in calculation, we regard upper left
    namely the cell[0,0] as the origin. However, the vector field derived by tensor field regards down left 
    as the origin, namely the cell[size[0]-1,0], only by this can the the value in vector field make sense.
    '''
    gamma[i] = gamma[i-1]+delta_t*gamma_dot[i-1]
#     gamma_dot[i] = vect_interp_2d(gamma[i-1,0], gamma[i-1,1], vector_lin)+delta_t*vect_interp_2d(gamma[i-1,0], gamma[i-1,1], dvv)
    gamma_dot[i] = gamma_dot[i-1]+delta_t*vect_interp_2d(gamma[i-1,0], gamma[i-1,1], dvv)
    
    if np.inner(-gamma_dot[i],gamma_dot[i-1])>np.inner(gamma_dot[i],gamma_dot[i-1]):
      gamma_dot[i] = -gamma_dot[i]
    
    if (math.ceil(gamma[i,0]) > 0 and math.ceil(gamma[i,0]) < vector_lin.shape[1]
        and math.ceil(gamma[i,1]) > 0 and math.ceil(gamma[i,1]) < vector_lin.shape[2]
        and mask_image[int(math.ceil(gamma[i,0])), int(math.ceil(gamma[i,1]))] > 0):
      points_x.append(gamma[i,0])
      points_y.append(gamma[i,1])
    else:
      break

  if both_directions:
#     points_x = points_x[::-1] + back_x
#     points_y = points_y[::-1] + back_y
    points_x = np.concatenate((points_x[::-1], back_x),0)
    points_y = np.concatenate((points_y[::-1], back_y),0)
    
  if filename:
    io.writePath2D(points_x, points_y, filename)

  return np.array(points_x), np.array(points_y)
