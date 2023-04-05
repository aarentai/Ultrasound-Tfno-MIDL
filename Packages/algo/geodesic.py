import math
from util.tensors import *
from lazy_imports import np
from lazy_imports import torch
from util import diff, maskops, riemann, tensors
from data import io
from tqdm import tqdm

from numba import jit
import pdb
# uncomment this for legit @profile when not using kernprof
def profile(blah):                
  return blah

def get_gamma_ddot_at_point(x, y, Gamma_field, gamma_dot):
  tens = tens_interp(x,y,Gamma_field)
  Gamma11 = tens[0,0]
  Gamma12 = tens[0,1]
  Gamma22 = tens[1,1]

  gamma_ddot = -(Gamma11*gamma_dot[0]*gamma_dot[0]
                 +Gamma12*gamma_dot[0]*gamma_dot[1]
                 +Gamma12*gamma_dot[1]*gamma_dot[0]
                 +Gamma22*gamma_dot[1]*gamma_dot[1])

  return(gamma_ddot)

def get_gamma_ddot_at_point_torch(x, y, Gamma_field, gamma_dot):
  tens = tens_interp_torch(x,y,Gamma_field).clone()
  Gamma11 = tens[0,0]
  Gamma12 = tens[0,1]
  Gamma22 = tens[1,1]

  gamma_ddot = -(Gamma11*gamma_dot[0]*gamma_dot[0]
                 +Gamma12*gamma_dot[0]*gamma_dot[1]
                 +Gamma12*gamma_dot[1]*gamma_dot[0]
                 +Gamma22*gamma_dot[1]*gamma_dot[1])
  #gamma_ddot = -term1 - term2 - term3 - term4

  return(gamma_ddot)

def batch_get_gamma_ddot_at_point_2d(x, y, Gamma_field, gamma_dot):
  tens = batch_tens_interp_2d(x,y,Gamma_field)
  Gamma11 = tens[:,0,0]
  Gamma12 = tens[:,0,1]
  Gamma22 = tens[:,1,1]

  gamma_ddot = -(Gamma11*gamma_dot[:,0]*gamma_dot[:,0]
                 +Gamma12*gamma_dot[:,0]*gamma_dot[:,1]
                 +Gamma12*gamma_dot[:,1]*gamma_dot[:,0]
                 +Gamma22*gamma_dot[:,1]*gamma_dot[:,1])

  return(gamma_ddot)

def batch_get_gamma_ddot_at_point_2d_torch(x, y, Gamma_field, gamma_dot):
  tens = batch_tens_interp_2d_torch(x,y,Gamma_field)
  Gamma11 = tens[:,0,0]
  Gamma12 = tens[:,0,1]
  Gamma22 = tens[:,1,1]

  gamma_ddot = -(Gamma11*gamma_dot[:,0]*gamma_dot[:,0]
                 +Gamma12*gamma_dot[:,0]*gamma_dot[:,1]
                 +Gamma12*gamma_dot[:,1]*gamma_dot[:,0]
                 +Gamma22*gamma_dot[:,1]*gamma_dot[:,1])

  return(gamma_ddot)

#@jit(nopython=True)
def batch_get_gamma_ddot_at_point_3d(x, y, z, Gamma_field, gamma_dot):
  tens = batch_tens_interp_3d(x,y,z,Gamma_field)
  Gamma11 = tens[:,0,0]
  Gamma12 = tens[:,0,1]
  Gamma13 = tens[:,0,2]
  Gamma22 = tens[:,1,1]
  Gamma23 = tens[:,1,2]
  Gamma33 = tens[:,2,2]

  gamma_ddot = -(Gamma11*gamma_dot[:,0]*gamma_dot[:,0]
                 +Gamma12*gamma_dot[:,0]*gamma_dot[:,1]
                 +Gamma12*gamma_dot[:,1]*gamma_dot[:,0]
                 +Gamma13*gamma_dot[:,0]*gamma_dot[:,2]
                 +Gamma13*gamma_dot[:,2]*gamma_dot[:,0]
                 +Gamma22*gamma_dot[:,1]*gamma_dot[:,1]
                 +Gamma23*gamma_dot[:,1]*gamma_dot[:,2]
                 +Gamma23*gamma_dot[:,2]*gamma_dot[:,1]
                 +Gamma33*gamma_dot[:,2]*gamma_dot[:,2])

  return(gamma_ddot)

def batch_get_gamma_ddot_at_point_3d_torch(x, y, z, Gamma_field, gamma_dot):
  tens = batch_tens_interp_3d_torch(x,y,z,Gamma_field)
  Gamma11 = tens[:,0,0]
  Gamma12 = tens[:,0,1]
  Gamma13 = tens[:,0,2]
  Gamma22 = tens[:,1,1]
  Gamma23 = tens[:,1,2]
  Gamma33 = tens[:,2,2]

  gamma_ddot = -(Gamma11*gamma_dot[:,0]*gamma_dot[:,0]
                 +Gamma12*gamma_dot[:,0]*gamma_dot[:,1]
                 +Gamma12*gamma_dot[:,1]*gamma_dot[:,0]
                 +Gamma13*gamma_dot[:,0]*gamma_dot[:,2]
                 +Gamma13*gamma_dot[:,2]*gamma_dot[:,0]
                 +Gamma22*gamma_dot[:,1]*gamma_dot[:,1]
                 +Gamma23*gamma_dot[:,1]*gamma_dot[:,2]
                 +Gamma23*gamma_dot[:,2]*gamma_dot[:,1]
                 +Gamma33*gamma_dot[:,2]*gamma_dot[:,2])

  return(gamma_ddot)

def geodesicpath(mode, tensor_lin, vector_lin, mask_image, start_coordinate, initial_velocity, delta_t=0.15, iter_num=18000, filename = '', both_directions=False):
  geodesicpath_points_x = np.zeros((iter_num))
  geodesicpath_points_y = np.zeros((iter_num))
  tensor_mat = lin2mat(tensor_lin)

  init_v = initial_velocity
  if initial_velocity is None:
    init_v = direction(start_coordinate, tensor_lin)

  if both_directions:
    back_x, back_y = geodesicpath(mode, tensor_lin, vector_lin, mask_image, start_coordinate, -init_v, delta_t, iter_num, filename, both_directions=False)

  print(f"Shooting geodesic path from {start_coordinate} with initial velocity {init_v}")

  metric_mat = np.linalg.inv(tensor_mat)
  DV = riemann.get_jacobian_2d(vector_lin, mask_image)
  dvv = np.einsum('...ij,j...->i...', DV, vector_lin)
  Gamma1, Gamma2 = riemann.get_christoffel_symbol_2d(metric_mat, mask_image)
  vgammav = np.zeros_like(vector_lin)
  vgammav[0] = np.einsum('i...,i...->...', vector_lin, np.einsum('...ij,j...->i...', Gamma1, vector_lin))
  vgammav[1] = np.einsum('i...,i...->...', vector_lin, np.einsum('...ij,j...->i...', Gamma2, vector_lin))
  nabla_vv = riemann.covariant_derivative_2d(vector_lin, metric_mat, mask_image)
  sigma = ((vector_lin[0]*nabla_vv[0]+vector_lin[1]*nabla_vv[1])/(vector_lin[0]**2+vector_lin[1]**2))
  sigmav = np.zeros_like(vector_lin)
  sigmav[0] = sigma*vector_lin[0]
  sigmav[1] = sigma*vector_lin[1]

  gamma = np.zeros((iter_num,2))
  gamma_dot = np.zeros((iter_num,2))
  gamma_ddot = np.zeros((iter_num,2))
  gamma_ddot_gt = np.zeros((iter_num,2))
  gamma[0] = start_coordinate
  gamma_dot[0] = init_v
  gamma_ddot[0, 0] = -np.einsum('i,i->', gamma_dot[0], np.einsum('ij,j->i',tens_interp_2d(gamma[0,0], gamma[0,1], Gamma1),gamma_dot[0]))
  gamma_ddot[0, 1] = -np.einsum('i,i->', gamma_dot[0], np.einsum('ij,j->i',tens_interp_2d(gamma[0,0], gamma[0,1], Gamma2),gamma_dot[0]))

  g_ddot = np.zeros((2))
  gamma[1] = gamma[0] +delta_t*gamma_dot[0]
    
  for i in range(2,iter_num):
    '''not exactly matched: gamma_ddot=-gamma_dot*\Gamma*gamma_dot'''
    if mode=='f':
      Gamma1_gamma = tens_interp_2d(gamma[i-2,0], gamma[i-2,1], Gamma1)
      Gamma2_gamma = tens_interp_2d(gamma[i-2,0], gamma[i-2,1], Gamma2)
      gamma_ddot[i-2,0] = -np.einsum('i,i->', gamma_dot[i-2], np.einsum('ij,j->i',Gamma1_gamma,gamma_dot[i-2]))
      gamma_ddot[i-2,1] = -np.einsum('i,i->', gamma_dot[i-2], np.einsum('ij,j->i',Gamma2_gamma,gamma_dot[i-2]))
      gamma_dot[i-1] = gamma_dot[i-2]+delta_t*(gamma_ddot[i-2])
      gamma[i] = gamma[i-1]+delta_t*gamma_dot[i-1]
        
    '''not exactly matched: gamma_ddot=-V*\Gamma*V+\sigma*V'''
    if mode=='g':
      Gamma1_gamma = tens_interp_2d(gamma[i-2,0], gamma[i-2,1], Gamma1)
      Gamma2_gamma = tens_interp_2d(gamma[i-2,0], gamma[i-2,1], Gamma2)
      gamma_ddot[i-2] = vect_interp_2d(gamma[i-2,0], gamma[i-2,1], -vgammav+sigmav)
      gamma_dot[i-1] = gamma_dot[i-2]+delta_t*(gamma_ddot[i-2])
      gamma[i] = gamma[i-1]+delta_t*gamma_dot[i-1]
        
    '''not exactly matched: gamma_ddot=-gamma_dot*\Gamma*gamma_dot+\sigma*gamma_dot'''
    if mode=='a':
      Gamma1_gamma = tens_interp_2d(gamma[i-2,0], gamma[i-2,1], Gamma1)
      Gamma2_gamma = tens_interp_2d(gamma[i-2,0], gamma[i-2,1], Gamma2)
      gamma_ddot[i-2,0] = -np.einsum('i,i->', gamma_dot[i-2], np.einsum('ij,j->i',Gamma1_gamma,gamma_dot[i-2]))
      gamma_ddot[i-2,1] = -np.einsum('i,i->', gamma_dot[i-2], np.einsum('ij,j->i',Gamma2_gamma,gamma_dot[i-2]))
      gamma_ddot[i-2] += image_interp_2d(gamma[i-2,0], gamma[i-2,1], sigma)*gamma_dot[i-2]
      gamma_dot[i-1] = gamma_dot[i-2]+delta_t*(gamma_ddot[i-2])
      gamma[i] = gamma[i-1]+delta_t*gamma_dot[i-1]
    
    '''not exactly matched: gamma_ddot=-gamma_dot*\Gamma*gamma_dot+\sigma*V'''
    if mode=='b':
      Gamma1_gamma = tens_interp_2d(gamma[i-2,0], gamma[i-2,1], Gamma1)
      Gamma2_gamma = tens_interp_2d(gamma[i-2,0], gamma[i-2,1], Gamma2)
      gamma_ddot[i-2,0] = -np.einsum('i,i->', gamma_dot[i-2], np.einsum('ij,j->i',Gamma1_gamma,gamma_dot[i-2]))
      gamma_ddot[i-2,1] = -np.einsum('i,i->', gamma_dot[i-2], np.einsum('ij,j->i',Gamma2_gamma,gamma_dot[i-2]))
      gamma_ddot[i-2] += vect_interp_2d(gamma[i-2,0], gamma[i-2,1], sigmav)
      gamma_dot[i-1] = gamma_dot[i-2]+delta_t*(gamma_ddot[i-2])
      gamma[i] = gamma[i-1]+delta_t*gamma_dot[i-1]

    '''not exactly matched: gamma_ddot=-gamma_dot*\Gamma*gamma_dot+\sigmaV+\epsilon'''
    if mode=='c':
      Gamma1_gamma = tens_interp_2d(gamma[i-2,0], gamma[i-2,1], Gamma1)
      Gamma2_gamma = tens_interp_2d(gamma[i-2,0], gamma[i-2,1], Gamma2)
      gamma_ddot[i-2,0] = -np.einsum('i,i->', gamma_dot[i-2], np.einsum('ij,j->i',Gamma1_gamma,gamma_dot[i-2]))
      gamma_ddot[i-2,1] = -np.einsum('i,i->', gamma_dot[i-2], np.einsum('ij,j->i',Gamma2_gamma,gamma_dot[i-2]))
      gamma_ddot[i-2] += vect_interp_2d(gamma[i-2,0], gamma[i-2,1], nabla_vv)
      gamma_dot[i-1] = gamma_dot[i-2]+delta_t*(gamma_ddot[i-2])
      gamma[i] = gamma[i-1]+delta_t*gamma_dot[i-1]

    '''not exactly matched: gamma_ddot=-V\GammaV+\sigmaV+\epsilon, with less interpolation'''
    if mode=='d':
      gamma_ddot[i-2] = vect_interp_2d(gamma[i-2,0], gamma[i-2,1], -vgammav+nabla_vv)
      gamma_dot[i-1] = gamma_dot[i-2]+delta_t*(gamma_ddot[i-2])
      gamma[i] = gamma[i-1]+delta_t*gamma_dot[i-1]
    
    '''exactly matched: gamma_ddot=DV*V, with less interpolation'''
    if mode=='e':
      gamma_ddot[i-2] = vect_interp_2d(gamma[i-2,0], gamma[i-2,1], dvv)
      gamma_dot[i-1] = gamma_dot[i-2]+delta_t*(gamma_ddot[i-2])
      gamma[i] = gamma[i-1]+delta_t*gamma_dot[i-1]
    
    if (math.ceil(gamma[i, 0]) > 0 and math.ceil(gamma[i, 0]) < vector_lin.shape[1]
       and math.ceil(gamma[i, 1]) > 0 and math.ceil(gamma[i, 1])  < vector_lin.shape[2]
       and mask_image[int(math.ceil(gamma[i, 0])), int(math.ceil(gamma[i, 1]))] > 0):
      geodesicpath_points_x[i-2] = gamma[i, 0]
      geodesicpath_points_y[i-2] = gamma[i, 1]
    else:
      # truncate and stop
      geodesicpath_points_x = geodesicpath_points_x[:i-2]
      geodesicpath_points_y = geodesicpath_points_y[:i-2]
      break

  if both_directions:
#     geodesicpath_points_x = torch.cat((torch.flip(geodesicpath_points_x,[0]), back_x),0)
#     geodesicpath_points_y = torch.cat((torch.flip(geodesicpath_points_y,[0]), back_y),0)
    geodesicpath_points_x = np.concatenate((geodesicpath_points_x[::-1], back_x),0)
    geodesicpath_points_y = np.concatenate((geodesicpath_points_y[::-1], back_y),0)
    
  if filename:
    io.writePath(geodesicpath_points_x, geodesicpath_points_y, filename)

  return geodesicpath_points_x, geodesicpath_points_y#, vgammav, dvv, nabla_vv

def batch_geodesicpath_2d(mode, tensor_lin, vector_lin, mask_image, start_coordinates, initial_velocities, delta_t=0.15, iter_num=18000, both_directions=False, Gamma1=None, Gamma2=None):
  # Compute 2d geodesic path of metric, where the input tensor_lin is the inverse of the metric
  # Assumes that mask_image is already a differentiable mask
  # To speed up when calling multiple times, precompute gammas by calling
  #  Gamma1, Gamma2 = compute_gammas_2d(tensor_lin, mask_image)
  # and pass in to method

  num_paths = len(start_coordinates)
  xsz,ysz = tensor_lin.shape[1:3]
  tensor_mat = lin2mat(tensor_lin)
  
  geodesicpath_points_x = [np.zeros((iter_num-1)) for p in range(num_paths)]
  geodesicpath_points_y = [np.zeros((iter_num-1)) for p in range(num_paths)]
  continue_path = np.ones((num_paths))
#   if ( (Gamma1 is None) or (Gamma2 is None)):
#     Gamma1, Gamma2 = compute_gammas_2d(tensor_lin, mask_image)
#   Gamma1 = tensors.lin2mat(Gamma1)
#   Gamma2 = tensors.lin2mat(Gamma2)
  metric_mat = np.linalg.inv(tensor_mat)
  Gamma1, Gamma2 = riemann.get_christoffel_symbol_2d(metric_mat, mask_image)

  init_v = initial_velocities

  if both_directions:
    back_x, back_y = batch_geodesicpath_2d(mode, tensor_lin, vector_lin, mask_image, start_coordinates,\
                                                   -init_v, delta_t, iter_num, both_directions=False,\
                                                   Gamma1=Gamma1, Gamma2=Gamma2)

  gamma = np.zeros((num_paths,iter_num,2))
  gamma_dot = np.zeros((num_paths,iter_num,2))
  gamma_ddot = np.zeros((num_paths,iter_num,2))
  gamma[:, 0] = start_coordinates
  gamma_dot[:, 0] = init_v
  gamma_ddot[:,0,0] = -np.einsum('...i,...i->...', gamma_dot[:,0], np.einsum('...ij,...j->...i',batch_tens_interp_2d(gamma[:,0,0], gamma[:,0,1], Gamma1),gamma_dot[:,0]))
  gamma_ddot[:,0,1] = -np.einsum('...i,...i->...', gamma_dot[:,0], np.einsum('...ij,...j->...i',batch_tens_interp_2d(gamma[:,0,0], gamma[:,0,1], Gamma2),gamma_dot[:,0]))
  gamma[:,1] = gamma[:,0] +delta_t*gamma_dot[:,0]

  metric_mat = np.linalg.inv(tensor_mat)
  DV = riemann.get_jacobian_2d(vector_lin, mask_image)
  dvv = np.einsum('...ij,j...->i...', DV, vector_lin)
  vgammav = np.zeros_like(vector_lin)
  vgammav[0] = np.einsum('i...,i...->...', vector_lin, np.einsum('...ij,j...->i...', Gamma1, vector_lin))
  vgammav[1] = np.einsum('i...,i...->...', vector_lin, np.einsum('...ij,j...->i...', Gamma2, vector_lin))
  nabla_vv = riemann.covariant_derivative_2d(vector_lin, metric_mat, mask_image)
  sigma = ((vector_lin[0]*nabla_vv[0]+vector_lin[1]*nabla_vv[1])/(vector_lin[0]**2+vector_lin[1]**2))
  sigmav = np.zeros_like(vector_lin)
  sigmav[0] = sigma*vector_lin[0]
  sigmav[1] = sigma*vector_lin[1]

  for i in tqdm(range(2,iter_num)):
    if mode=='f':
      Gamma1_gamma = batch_tens_interp_2d(gamma[:,i-2,0], gamma[:,i-2,1], Gamma1)
      Gamma2_gamma = batch_tens_interp_2d(gamma[:,i-2,0], gamma[:,i-2,1], Gamma2)
      gamma_ddot[:,i-2,0] = -np.einsum('...i,...i->...', gamma_dot[:,i-2], np.einsum('...ij,...j->...i',Gamma1_gamma,gamma_dot[:,i-2]))
      gamma_ddot[:,i-2,1] = -np.einsum('...i,...i->...', gamma_dot[:,i-2], np.einsum('...ij,...j->...i',Gamma2_gamma,gamma_dot[:,i-2]))
      gamma_dot[:,i-1] = gamma_dot[:,i-2]+delta_t*(gamma_ddot[:,i-2])
      gamma[:,i] = gamma[:,i-1]+delta_t*gamma_dot[:,i-1]
        
    '''not exactly matched: gamma_ddot=-V*\Gamma*V+\sigma*V'''
    if mode=='g':
      Gamma1_gamma = batch_tens_interp_2d(gamma[:,i-2,0], gamma[:,i-2,1], Gamma1)
      Gamma2_gamma = batch_tens_interp_2d(gamma[:,i-2,0], gamma[:,i-2,1], Gamma2)
#       gamma_ddot[:,i-2,0] = -np.einsum('...i,...i->...', batch_vect_interp_2d(gamma[:,i-2,0], gamma[:,i-2,1], vector_lin), np.einsum('...ij,...j->...i',Gamma1_gamma,vect_interp_2d(gamma[:,i-2,0], gamma[:,i-2,1], vector_lin)))
#       gamma_ddot[:,i-2,1] = -np.einsum('...i,...i->...', batch_vect_interp_2d(gamma[:,i-2,0], gamma[:,i-2,1], vector_lin), np.einsum('...ij,...j->...i',Gamma2_gamma,vect_interp_2d(gamma[:,i-2,0], gamma[:,i-2,1], vector_lin)))
      gamma_ddot[:,i-2] = batch_vect_interp_2d(gamma[:,i-2,0], gamma[:,i-2,1], -vgammav+sigmav)
      gamma_dot[:,i-1] = gamma_dot[:,i-2]+delta_t*(gamma_ddot[:,i-2])
      gamma[:,i] = gamma[:,i-1]+delta_t*gamma_dot[:,i-1]
        
    '''not exactly matched: gamma_ddot=-gamma_dot*\GammaV*gamma_dot+\sigma*gamma_dot'''
    if mode=='a':
      Gamma1_gamma = batch_tens_interp_2d(gamma[:,i-2,0], gamma[:,i-2,1], Gamma1)
      Gamma2_gamma = batch_tens_interp_2d(gamma[:,i-2,0], gamma[:,i-2,1], Gamma2)
      gamma_ddot[:,i-2,0] = -np.einsum('...i,...i->...', gamma_dot[:,i-2], np.einsum('...ij,...j->...i',Gamma1_gamma,gamma_dot[:,i-2]))
      gamma_ddot[:,i-2,1] = -np.einsum('...i,...i->...', gamma_dot[:,i-2], np.einsum('...ij,...j->...i',Gamma2_gamma,gamma_dot[:,i-2]))
      gamma_ddot[:,i-2] += batch_image_interp_2d(gamma[:,i-2,0], gamma[:,i-2,1], sigma)*gamma_dot[:,i-2]
      gamma_dot[:,i-1] = gamma_dot[:,i-2]+delta_t*(gamma_ddot[:,i-2])
      gamma[:,i] = gamma[:,i-1]+delta_t*gamma_dot[:,i-1]
    
    '''not exactly matched: gamma_ddot=-gamma_dot*\GammaV*gamma_dot+\sigma*V'''
    if mode=='b':
      Gamma1_gamma = batch_tens_interp_2d(gamma[:,i-2,0], gamma[:,i-2,1], Gamma1)
      Gamma2_gamma = batch_tens_interp_2d(gamma[:,i-2,0], gamma[:,i-2,1], Gamma2)
      gamma_ddot[:,i-2,0] = -np.einsum('...i,...i->...', gamma_dot[:,i-2], np.einsum('...ij,...j->...i',Gamma1_gamma,gamma_dot[:,i-2]))
      gamma_ddot[:,i-2,1] = -np.einsum('...i,...i->...', gamma_dot[:,i-2], np.einsum('...ij,...j->...i',Gamma2_gamma,gamma_dot[:,i-2]))
      gamma_ddot[:,i-2] += batch_vect_interp_2d(gamma[:,i-2,0], gamma[:,i-2,1], sigmav)
      gamma_dot[:,i-1] = gamma_dot[:,i-2]+delta_t*(gamma_ddot[:,i-2])
      gamma[:,i] = gamma[:,i-1]+delta_t*gamma_dot[:,i-1]
#       pdb.set_trace()

    '''not exactly matched: gamma_ddot=-gamma_dot*\GammaV*gamma_dot+\sigmaV+\epsilon'''
    if mode=='c':
      Gamma1_gamma = batch_tens_interp_2d(gamma[:,i-2,0], gamma[:,i-2,1], Gamma1)
      Gamma2_gamma = batch_tens_interp_2d(gamma[:,i-2,0], gamma[:,i-2,1], Gamma2)
      gamma_ddot[:,i-2,0] = -np.einsum('...i,...i->...', gamma_dot[:,i-2], np.einsum('...ij,...j->...i',Gamma1_gamma,gamma_dot[:,i-2]))
      gamma_ddot[:,i-2,1] = -np.einsum('...i,...i->...', gamma_dot[:,i-2], np.einsum('...ij,...j->...i',Gamma2_gamma,gamma_dot[:,i-2]))
      gamma_ddot[:,i-2] += batch_vect_interp_2d(gamma[:,i-2,0], gamma[:,i-2,1], nabla_vv)
      gamma_dot[:,i-1] = gamma_dot[:,i-2]+delta_t*(gamma_ddot[:,i-2])
      gamma[:,i] = gamma[:,i-1]+delta_t*gamma_dot[:,i-1]

    '''not exactly matched: gamma_ddot=-V\GammaV+\sigmaV+\epsilon'''
    if mode=='d':
      Gamma1_gamma = batch_tens_interp_2d(gamma[:,i-2,0], gamma[:,i-2,1], Gamma1)
      Gamma2_gamma = batch_tens_interp_2d(gamma[:,i-2,0], gamma[:,i-2,1], Gamma2)
#       v_gamma = batch_vect_interp_2d(gamma[:,i-2,0], gamma[:,i-2,1], vector_lin)
#       gamma_ddot[:,i-2,0] = -np.einsum('...i,...i->...', v_gamma, np.einsum('...ij,...j->...i', Gamma1_gamma, v_gamma))
#       gamma_ddot[:,i-2,1] = -np.einsum('...i,...i->...', v_gamma, np.einsum('...ij,...j->...i', Gamma2_gamma, v_gamma))
#       gamma_ddot[:,i-2] += vect_interp_2d(gamma[:,i-2,0], gamma[:,i-2,1], nabla_vv)
      gamma_ddot[:,i-2] = vect_interp_2d(gamma[:,i-2,0], gamma[:,i-2,1], -vgammav+nabla_vv)
      gamma_dot[:,i-1] = gamma_dot[:,i-2]+delta_t*(gamma_ddot[:,i-2])
      gamma[:,i] = gamma[:,i-1]+delta_t*gamma_dot[:,i-1]
    
    '''exactly matched: gamma_ddot=DV*V'''
    if mode=='e':
#       v_gamma = batch_vect_interp_2d(gamma[:,i-2,0], gamma[:,i-2,1], vector_lin)
#       DV_gamma = batch_tens_interp_2d(gamma[:,i-2,0], gamma[:,i-2,1], DV)
#       gamma_ddot[:,i-2] = np.einsum('...ij,...j->...i', DV_gamma, v_gamma)
      gamma_ddot[:,i-2] = batch_vect_interp_2d(gamma[:,i-2,0], gamma[:,i-2,1], dvv)
      gamma_dot[:,i-1] = gamma_dot[:,i-2]+delta_t*(gamma_ddot[:,i-2])
      gamma[:,i] = gamma[:,i-1]+delta_t*gamma_dot[:,i-1]
    
    active_path=False
    
    for p in range(num_paths):
#       print( continue_path[p]\
#        ,np.ceil(gamma[p, i, 0]) >= 0 and np.ceil(gamma[p, i, 0]) < xsz\
#        ,np.ceil(gamma[p, i, 1]) >= 0 and np.ceil(gamma[p, i, 1]) < ysz\
#        ,(gamma[p, i, 0], gamma[p, i, 1]))
      if ( continue_path[p]
       and np.ceil(gamma[p, i, 0]) >= 0 and np.ceil(gamma[p, i, 0]) < xsz
       and np.ceil(gamma[p, i, 1]) >= 0 and np.ceil(gamma[p, i, 1]) < ysz
       and (mask_image[int(np.ceil(gamma[p, i, 0])), int(np.ceil(gamma[p, i, 1]))] > 0)):# and gamma[p, i, 0]!=0 and gamma[p, i, 0]!=0:
        active_path = True
        geodesicpath_points_x[p][i-2] = gamma[p, i, 0]
        geodesicpath_points_y[p][i-2] = gamma[p, i, 1]
      else:
        # truncate and stop
#         geodesicpath_points_x[p] = geodesicpath_points_x[p][1:i-3]
#         geodesicpath_points_y[p] = geodesicpath_points_y[p][1:i-3]
        geodesicpath_points_x[p] = geodesicpath_points_x[p][:i-2]
        geodesicpath_points_y[p] = geodesicpath_points_y[p][:i-2]
        continue_path[p] = 0
    if not active_path:
      break
  # End for each time point i

  if both_directions:
    for p in range(num_paths):
      geodesicpath_points_x[p] = np.concatenate((geodesicpath_points_x[p][::-1], back_x[p]),0)
      geodesicpath_points_y[p] = np.concatenate((geodesicpath_points_y[p][::-1], back_y[p]),0)
    
  return geodesicpath_points_x, geodesicpath_points_y

def geodesicpath_hamilton(mode, tensor_lin, mask_image, start_coordinate, initial_velocity, delta_t=0.15, iter_num=18000, both_directions=False):
  geodesicpath_points_x = np.zeros((iter_num))
  geodesicpath_points_y = np.zeros((iter_num))

  init_v = initial_velocity
  if initial_velocity is None:
    init_v = direction(start_coordinate, tensor_lin)

  if both_directions:
    back_x, back_y = geodesicpath_hamilton(mode, tensor_lin, mask_image, start_coordinate, -init_v, delta_t, iter_num, both_directions=False)

  print(f"Finding geodesic path from {start_coordinate} with initial velocity {init_v}")
    
  tensor_mat = tensors.lin2mat(tensor_lin)
  metric_mat = np.linalg.inv(tensor_mat)

  d1_eps11, d2_eps11 = diff.gradient_mask_2d(tensor_mat[...,0,0], mask_image)
  d1_eps12, d2_eps12 = diff.gradient_mask_2d(tensor_mat[...,0,1], mask_image)
  d1_eps21, d2_eps21 = diff.gradient_mask_2d(tensor_mat[...,1,0], mask_image)
  d1_eps22, d2_eps22 = diff.gradient_mask_2d(tensor_mat[...,1,1], mask_image)

  d1_tensor_mat = np.zeros_like(tensor_mat)
  d1_tensor_mat[...,0,0] = d1_eps11
  d1_tensor_mat[...,0,1] = d1_eps12
  d1_tensor_mat[...,1,0] = d1_eps21
  d1_tensor_mat[...,1,1] = d1_eps22
    
  d2_tensor_mat = np.zeros_like(tensor_mat)
  d2_tensor_mat[...,0,0] = d2_eps11
  d2_tensor_mat[...,0,1] = d2_eps12
  d2_tensor_mat[...,1,0] = d2_eps21
  d2_tensor_mat[...,1,1] = d2_eps22

  gamma = np.zeros((iter_num,2))
  gamma[0] = start_coordinate
  p = np.zeros((iter_num,2))
  p[0] = np.einsum('ij,j->i', tens_interp_2d(gamma[0,0], gamma[0,1], metric_mat), init_v)
  
#   gamma[1] = gamma[0]+delta_t*gamma_dot[0]
  
  for i in range(1,iter_num):
    increment = np.einsum('ij,j->i', tens_interp_2d(gamma[i-1,0], gamma[i-1,1], tensor_mat), p[i-1])
#     print(increment)
    gamma[i] = gamma[i-1]+delta_t*increment
    if mode=='i-1':
      p[i,0] = p[i-1,0]-0.5*delta_t*np.einsum('i,i->', p[i-1], np.einsum('ij,j->i', tens_interp_2d(gamma[i-1,0], gamma[i-1,1], d1_tensor_mat), p[i-1]))
      p[i,1] = p[i-1,1]-0.5*delta_t*np.einsum('i,i->', p[i-1], np.einsum('ij,j->i', tens_interp_2d(gamma[i-1,0], gamma[i-1,1], d2_tensor_mat), p[i-1]))
    
    if mode=='i':
      p[i,0] = p[i-1,0]-0.5*delta_t*np.einsum('i,i->', p[i-1], np.einsum('ij,j->i', tens_interp_2d(gamma[i,0], gamma[i,1], d1_tensor_mat), p[i-1]))
      p[i,1] = p[i-1,1]-0.5*delta_t*np.einsum('i,i->', p[i-1], np.einsum('ij,j->i', tens_interp_2d(gamma[i,0], gamma[i,1], d2_tensor_mat), p[i-1]))
    
    if mode=='mean':
      p[i,0] = p[i-1,0]-0.5*delta_t*np.einsum('i,i->', p[i-1], np.einsum('ij,j->i', tens_interp_2d((gamma[i-1,0]+gamma[i,0])/2, (gamma[i-1,1]+gamma[i,1])/2, d1_tensor_mat), p[i-1]))
      p[i,1] = p[i-1,1]-0.5*delta_t*np.einsum('i,i->', p[i-1], np.einsum('ij,j->i', tens_interp_2d((gamma[i-1,0]+gamma[i,0])/2, (gamma[i-1,1]+gamma[i,1])/2, d2_tensor_mat), p[i-1]))
    
    if (math.ceil(gamma[i, 0]) > 0 and math.ceil(gamma[i, 0]) < d1_eps11.shape[0]
       and math.ceil(gamma[i, 1]) > 0 and math.ceil(gamma[i, 1])  < d1_eps11.shape[1]
       and mask_image[int(math.ceil(gamma[i, 0])), int(math.ceil(gamma[i, 1]))] > 0):
      geodesicpath_points_x[i-1] = gamma[i, 0]
      geodesicpath_points_y[i-1] = gamma[i, 1]
    else:
      # truncate and stop
      geodesicpath_points_x = geodesicpath_points_x[:i-1]
      geodesicpath_points_y = geodesicpath_points_y[:i-1]
      break
  if both_directions:
    geodesicpath_points_x = np.concatenate((geodesicpath_points_x[::-1], back_x),0)
    geodesicpath_points_y = np.concatenate((geodesicpath_points_y[::-1], back_y),0)

  return geodesicpath_points_x, geodesicpath_points_y

def compute_gammas_2d(tensor_field, mask_image):
      # Compute the Christoffel symbols Gamma1, Gamma2
  eps11 = tensor_field[0, :, :]
  eps12 = tensor_field[1, :, :]
  eps22 = tensor_field[2, :, :]

  eps_11 = eps22 / (eps11 * eps22 - eps12 ** 2)
  eps_12 = -eps12 / (eps11 * eps22 - eps12 ** 2)
  eps_22 = eps11 / (eps11 * eps22 - eps12 ** 2)

  bdry_type, bdry_idx, bdry_map = maskops.determine_boundary_2d(mask_image)
  d1_eps_11, d2_eps_11 = diff.gradient_bdry_2d(eps_11, bdry_idx, bdry_map)
  d1_eps_12, d2_eps_12 = diff.gradient_bdry_2d(eps_12, bdry_idx, bdry_map)
  d1_eps_22, d2_eps_22 = diff.gradient_bdry_2d(eps_22, bdry_idx, bdry_map)

  Gamma1_11 = (eps11 * d1_eps_11 + eps12 * (2 * d1_eps_12 - d2_eps_11)) / 2
  Gamma1_12 = (eps11 * d2_eps_11 + eps12 * d1_eps_22) / 2
  Gamma1_22 = (eps11 * (2 * d2_eps_12 - d1_eps_22) + eps12 * d2_eps_22) / 2
  Gamma1 = np.stack((Gamma1_11, Gamma1_12, Gamma1_22))
  Gamma2_11 = (eps12 * d1_eps_11 + eps22 * (2 * d1_eps_12 - d2_eps_11)) / 2
  Gamma2_12 = (eps12 * d2_eps_11 + eps22 * d1_eps_22) / 2
  Gamma2_22 = (eps12 * (2 * d2_eps_12 - d1_eps_22) + eps22 * d2_eps_22) / 2
  Gamma2 = np.stack((Gamma2_11, Gamma2_12, Gamma2_22))

  return (Gamma1, Gamma2)

def geodesic_between_points_torch(tensor_field, mask_image, start_coordinate, end_coordinate, init_velocity=None, step_size=0.0001, num_iters=18000, filename = ''):
  # assumes tensor_field and mask_image are np arrays, converts to torch here
  torch_field = torch.from_numpy(tensor_field)
  mask = torch.from_numpy(mask_image)
  start_coords = torch.tensor(start_coordinate)
  end_coords = torch.tensor(end_coordinate)

  # TODO Is there a way to use pytorch batching to compute many geodesics at once?
  energy = torch.zeros((num_iters))
  init_v = torch.zeros((num_iters, 2), requires_grad=True)
  
  if init_velocity is None:
    init_v[0] = direction_torch(start_coordinate, tensor_field)
  else:
    init_v[0] = torch.tensor(init_velocity)


  all_points_x = []
  all_points_y = []
  
  for it in range(0,num_iters-1):
    end_point, points_x, points_y = geodesicpath_torch(torch_field, mask, start_coords, init_v[it], delta_t=0.15, iter_num=18000, filename = '')
    all_points_x.append(points_x)
    all_points_y.append(points_y)
    energy[it] = ((end_point[0] - end_coords[0])**2 + (end_point[1] - end_coords[1])**2)
    energy.backward()
    init_v[it+1] = init_v[it] - step_size * init_v.grad

  return(all_points_x, all_points_y, init_v, energy)
