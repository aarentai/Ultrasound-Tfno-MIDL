import math
from lazy_imports import np
from lazy_imports import torch
from lazy_imports import sitk
from data.convert import get_framework, GetNPArrayFromSITK, GetSITKImageFromNP
from numba import jit, njit, prange


def mat2lin(mat):
    """
    transform [h,w,2,2]/[h,w,d,3,3] tensor to [3,h,w]/[6,h,w,d]
    Args:
        mat, torch.Tensor or np.array
    Returns: 
        lin, torch.Tensor or np.array
    """
    assert mat.shape[-1]==mat.shape[-2], 'Last two dimensions of input tensor should equal.'
    assert len(mat.shape)==mat.shape[-1]+2, 'Tensor\'s shape should follow [d1, ..., dn, n, n].'
    fw, fw_name = get_framework(mat)
    
    dim = len(mat.shape)-2
    lin = fw.zeros((dim*(dim+1)//2, *mat.shape[:dim]))
    k = 0
    
    for i in range(dim):
        for j in range(i, dim):
            lin[k] = mat[..., i, j]
            k += 1
    
    return lin

def lin2mat(lin):
    """
    transform [3,h,w]/[6,h,w,d] tensor to [h,w,2,2]/[h,w,d,3,3]
    Args:
        lin, torch.Tensor or np.array
    Returns: 
        mat, torch.Tensor or np.array
    """
    assert len(lin.shape)*(len(lin.shape)-1)/2==lin.shape[0], 'Tensor\'s shape should follow [n(n+1)/2, d1, ..., dn].'
    fw, fw_name = get_framework(lin)
    
    dim = len(lin.shape)-1
    mat = fw.zeros((*lin.shape[1:], dim, dim))
    k = 0
    
    for i in range(dim):
        for j in range(i, dim):
            mat[...,i,j] = lin[k]
            if i!=j:
                mat[...,j,i] = lin[k]
            k += 1
    
    return mat

def mat2lin_batch(mat):
    """
    transform [b,h,w,2,2]/[b,h,w,d,3,3] tensor to [b,3,h,w]/[b,6,h,w,d]
    Args:
        mat, torch.Tensor or np.array
    Returns: 
        lin, torch.Tensor or np.array
    """
    assert mat.shape[-1]==mat.shape[-2], 'Last two dimensions of input tensor should equal.'
    assert len(mat.shape)-3==mat.shape[-1], 'Tensor\'s shape should follow [d1, ..., dn, n, n].'
    fw, fw_name = get_framework(mat)
    
    dim = len(mat.shape)-3
    lin = fw.zeros((mat.shape[0], dim*(dim+1)//2, *mat.shape[:dim]))
    k = 0
    
    for i in range(dim):
        for j in range(i, dim):
            lin[:,k] = mat[..., i, j]
            k += 1
    
    return lin

def lin2mat_batch(lin):
    """
    transform [b,3,h,w]/[b,6,h,w,d] tensor to [b,h,w,2,2]/[b,h,w,d,3,3]
    Args:
        lin, torch.Tensor or np.array
    Returns: 
        mat, torch.Tensor or np.array
    """
    assert (len(lin.shape)-1)*(len(lin.shape)-2)/2==lin.shape[1], 'Tensor\'s shape should follow [n(n+1)/2, d1, ..., dn].'
    fw, fw_name = get_framework(lin)
    
    dim = len(lin.shape)-2
    mat = fw.zeros((lin.shape[0], *lin.shape[2:], dim, dim))
    k = 0
    
    for i in range(dim):
        for j in range(i, dim):
            mat[...,i,j] = lin[:,k]
            if i!=j:
                mat[...,j,i] = lin[:,k]
            k += 1
    
    return mat

def profile(blah):                
  return blah

def batch_cholesky_v2(tens):
  fw, fw_name = get_framework(tens)
  if fw_name == 'numpy':
    nan = fw.nan
  else:
    nan = fw.tensor(float('nan'))
  L = fw.zeros_like(tens)
  for xx in range(tens.shape[0]):
    for yy in range(tens.shape[1]):
      for zz in range(tens.shape[2]):
        try:
          L[xx,yy,zz] = fw.linalg.cholesky(tens[xx,yy,zz])
        except:
          L[xx,yy,zz] = nan * fw.ones((tens.shape[-2:]))
  return L

def batch_cholesky(tens):
  # from https://stackoverflow.com/questions/60230464/pytorch-torch-cholesky-ignoring-exception
  # will get NaNs instead of exception where cholesky is invalid
  fw, fw_name = get_framework(tens)
  L = fw.zeros_like(tens)

  for i in range(tens.shape[-1]):
    for j in range(i+1):
      s = 0.0
      for k in range(j):
        s = s + L[...,i,k] * L[...,j,k]

      L[...,i,j] = fw.sqrt(tens[...,i,i] - s) if (i == j) else \
                      (1.0 / L[...,j,j] * (tens[...,i,j] - s))
  return L

def smooth_tensors(tens, sigma):
  fw, fw_name = get_framework(tens)
  if fw_name == 'numpy':
    filt_tens = GetNPArrayFromSITK(
                sitk.RecursiveGaussian(sitk.RecursiveGaussian(sitk.RecursiveGaussian(
                  GetSITKImageFromNP(tens,True), sigma=sigma,direction=0), sigma=sigma,direction=1), sigma=sigma, direction=2),True)
  else:
    filt_tens = torch.from_numpy(GetNPArrayFromSITK(
                sitk.RecursiveGaussian(sitk.RecursiveGaussian(sitk.RecursiveGaussian(
                GetSITKImageFromNP(tens.cpu().numpy(),True), sigma=sigma,direction=0), sigma=sigma,direction=1), sigma=sigma,direction=2),True))
  return(filt_tens)

def tens_6_to_tens_3x3(tens):
  tens_full = np.zeros((tens.shape[0], tens.shape[1], tens.shape[2], 3, 3))
  tens_full[:,:,:,0,0] = tens[:,:,:,0]
  tens_full[:,:,:,0,1] = tens[:,:,:,1]
  tens_full[:,:,:,1,0] = tens[:,:,:,1]
  tens_full[:,:,:,0,2] = tens[:,:,:,2]
  tens_full[:,:,:,2,0] = tens[:,:,:,2]
  tens_full[:,:,:,1,1] = tens[:,:,:,3]
  tens_full[:,:,:,1,2] = tens[:,:,:,4]
  tens_full[:,:,:,2,1] = tens[:,:,:,4]
  tens_full[:,:,:,2,2] = tens[:,:,:,5]
  return(tens_full)

def tens_3x3_to_tens_6(tens):
  tens_tri = np.zeros((tens.shape[0], tens.shape[1], tens.shape[2], 6))
  tens_tri[:,:,:,0] = tens[:,:,:,0,0]
  tens_tri[:,:,:,1] = tens[:,:,:,0,1]
  tens_tri[:,:,:,2] = tens[:,:,:,0,2]
  tens_tri[:,:,:,3] = tens[:,:,:,1,1]
  tens_tri[:,:,:,4] = tens[:,:,:,1,2]
  tens_tri[:,:,:,5] = tens[:,:,:,2,2]
  return(tens_tri)

def direction(coordinate, tensor_field):
  tens = tens_interp(coordinate[0], coordinate[1], tensor_field)
  u, v = eigen_vec(tens)
  return (np.array([u, v]))

def direction_torch(coordinate, tensor_field):
  tens = tens_interp_torch(coordinate[0], coordinate[1], tensor_field)
  u, v = eigen_vec_torch(tens)
  return (torch.tensor([u, v]))

def direction_3d(coordinate, tensor_field):
  tens = tens_interp_3d(coordinate[0], coordinate[1], coordinate[2], tensor_field)
  u, v, w = eigen_vec_3d(tens)
  return (np.array([u, v, w]))

def direction_3d_torch(coordinate, tensor_field):
  tens = tens_interp_3d_torch(coordinate[0], coordinate[1], coordinate[2], tensor_field)
  u, v, w = eigen_vec_3d_torch(tens)
  return (torch.tensor((u, v, w)))

def batch_direction_2d(coordinates, tensor_field):
  tens = batch_tens_interp_2d(coordinates[:,0], coordinates[:,1], tensor_field)
  directions = batch_eigen_vec_2d(tens)
  return (directions)

def batch_direction_2d_torch(coordinates, tensor_field):
  tens = batch_tens_interp_2d_torch(coordinates[:,0], coordinates[:,1], tensor_field)
  directions = batch_eigen_vec_2d_torch(tens)
  return (directions)

#@jit(nopython=True)
def batch_direction_3d(coordinates, tensor_field):
  tens = batch_tens_interp_3d(coordinates[:,0], coordinates[:,1], coordinates[:,2], tensor_field)
  directions = batch_eigen_vec_3d(tens)
  return (directions)

def batch_direction_3d_torch(coordinates, tensor_field):
  tens = batch_tens_interp_3d_torch(coordinates[:,0], coordinates[:,1], coordinates[:,2], tensor_field)
  directions = batch_eigen_vec_3d_torch(tens)
  return (directions)

def fractional_anisotropy(g):
    e, _ = torch.symeig(g)
    lambd1 = e[:,:,:,0]
    lambd2 = e[:,:,:,1]
    lambd3 = e[:,:,:,2]
    mean = torch.mean(e,dim=len(e.shape)-1)
    return torch.sqrt(3.*(torch.pow((lambd1-mean),2)+torch.pow((lambd2-mean),2)+torch.pow((lambd3-mean),2)))/\
    torch.sqrt(2.*(torch.pow(lambd1,2)+torch.pow(lambd2,2)+torch.pow(lambd3,2)))

def eigen_vec(tens):
  evals, evecs = np.linalg.eigh(tens)
  [u, v] = evecs[:, evals.argmax()]
  return (u, v)

def eigen_vec_torch(tens):
  evals, evecs =torch.symeig(tens, eigenvectors=True)
  [u, v] = evecs[:, evals.argmax()]
  return (u, v)

def eigen_vec_3d(tens):
  evals, evecs = np.linalg.eigh(tens)
  [u, v, w] = evecs[:, evals.argmax()]
  return (u, v, w)

def eigen_vec_3d_torch(tens):
  #evals, evecs = torch.symeig(tens,eigenvectors=True)
  #evals, evecs = torch.linalg.eigh(tens)
  #[u, v, w] = evecs[:, evals.argmax()]
  evals, evecs = torch_sym3eig.Sym3Eig.apply(tens.reshape((-1,3,3)))
  [u, v, w] = evecs[0, :, evals.argmax()]
  return (u,v,w)

#@jit(nopython=True)
def batch_eigen_vec_3d(tens):
  evals, evecs = np.linalg.eigh(tens)
  #return (evecs[:, :, evals.argmax(axis=1)])
  idx = np.expand_dims(np.expand_dims(evals.argmax(axis=1),axis=-1),axis=-1)
  return(np.take_along_axis(evecs, idx, axis=2).reshape((-1,3)))

def batch_eigen_vec_3d_torch(tens):
  #evals, evecs = torch.symeig(tens,eigenvectors=True)
  #evals, evecs = torch.linalg.eigh(tens)
  #[u, v, w] = evecs[:, evals.argmax()]
  evals, evecs = torch_sym3eig.Sym3Eig.apply(tens.reshape((-1,3,3)))
  #return (evecs[:, :, evals.argmax(axis=1)])
  idx = torch.unsqueeze(torch.unsqueeze(torch.argmax(evals, dim=1),-1),-1)
  return(torch.take_along_dim(evecs, idx, dim = 2).reshape((-1,3)))

def circ_shift(I, shift):
  I = np.roll(I, shift[0], axis=0)
  I = np.roll(I, shift[1], axis=1)
  return (I)

def circ_shift_torch(I, shift):
  I = torch.roll(I, shift[0], dims=0)
  I = torch.roll(I, shift[1], dims=1)
  return (I)

def circ_shift_3d(I, shift):
  I = np.roll(I, shift[0], axis=0)
  I = np.roll(I, shift[1], axis=1)
  I = np.roll(I, shift[2], axis=2)
  return (I)

def circ_shift_3d_torch(I, shift):
  I = torch.roll(I, shift[0], dims=0)
  I = torch.roll(I, shift[1], dims=1)
  I = torch.roll(I, shift[2], dims=2)
  return (I)

def image_interp_2d(x, y, image):
  scalar = 0.0
  if ((math.floor(x) < 0) or (math.ceil(x) >= image.shape[0])
      or (math.floor(y) < 0) or (math.ceil(y) >= image.shape[1])):
    # data is out of bounds, return identity vector
    scalar = 0.0
    return(scalar)
   
  if x == math.floor(x) and y == math.floor(y):
    scalar = image[int(x), int(y)]
  elif x != math.floor(x) and y == math.floor(y):
    scalar = abs(x - math.floor(x)) * image[math.ceil(x), int(y)] \
                 + abs(x - math.ceil(x)) * image[math.floor(x), int(y)]
  elif x == math.floor(x) and y != math.floor(y):
    scalar = abs(y - math.floor(y)) * image[int(x), math.ceil(y)] \
                 + abs(y - math.ceil(y)) * image[int(x), math.floor(y)]
  else:
    scalar = abs(x - math.floor(x)) * abs(y - math.floor(y)) * image[math.ceil(x), math.ceil(y)] \
           + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * image[math.ceil(x), math.floor(y)] \
           + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * image[math.floor(x), math.ceil(y)] \
           + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * image[math.floor(x), math.floor(y)] 
  return scalar

def batch_image_interp_2d(x, y, image):
  num_tens = x.shape[0]
  scalar = np.zeros((num_tens, 1))
  for p in range(num_tens):
    if ((math.floor(x[p]) < 0) or (math.ceil(x[p]) >= image.shape[0]) or (math.floor(y[p]) < 0) or (math.ceil(y[p]) >= image.shape[1])):
      # data is out of bounds, return identity vector
      scalar[p] = 0.0
      continue

    if x[p] == math.floor(x[p]) and y[p] == math.floor(y[p]):
      scalar[p] = image[int(x[p]), int(y[p])]
    elif x[p] != math.floor(x[p]) and y[p] == math.floor(y[p]):
      scalar[p] = abs(x[p] - math.floor(x[p])) * image[math.ceil(x[p]), int(y[p])] \
                     + abs(x[p] - math.ceil(x[p])) * image[math.floor(x[p]), int(y[p])]
    elif x[p] == math.floor(x[p]) and y != math.floor(y[p]):
      scalar[p] = abs(y[p] - math.floor(y[p])) * image[int(x[p]), math.ceil(y[p])] \
                     + abs(y[p] - math.ceil(y[p])) * image[int(x[p]), math.floor(y[p])]
    else:
      scalar[p] = abs(x[p] - math.floor(x[p])) * abs(y[p] - math.floor(y[p])) * image[math.ceil(x[p]), math.ceil(y[p])] \
               + abs(x[p] - math.floor(x[p])) * abs(y[p] - math.ceil(y[p])) * image[math.ceil(x[p]), math.floor(y[p])] \
               + abs(x[p] - math.ceil(x[p])) * abs(y[p] - math.floor(y[p])) * image[math.floor(x[p]), math.ceil(y[p])] \
               + abs(x[p] - math.ceil(x[p])) * abs(y[p] - math.ceil(y[p])) * image[math.floor(x[p]), math.floor(y[p])] 
  return scalar

def vect_interp_2d(x, y, vector_field):
  vect = np.zeros((2))
  v1 = vector_field[0]
  v2 = vector_field[1]
  if ((math.floor(x) < 0) or (math.ceil(x) >= v1.shape[0])
      or (math.floor(y) < 0) or (math.ceil(y) >= v1.shape[1])):
    # data is out of bounds, return identity vector
    vect[0] = 1
    vect[1] = 1
    return(vect)
   
  if x == math.floor(x) and y == math.floor(y):
    vect[0] = v1[int(x), int(y)]
    vect[1] = v2[int(x), int(y)]
  elif x != math.floor(x) and y == math.floor(y):
    vect[0] = abs(x - math.floor(x)) * v1[math.ceil(x), int(y)] \
                 + abs(x - math.ceil(x)) * v1[math.floor(x), int(y)]
    vect[1] = abs(x - math.floor(x)) * v2[math.ceil(x), int(y)] \
                 + abs(x - math.ceil(x)) * v2[math.floor(x), int(y)]
  elif x == math.floor(x) and y != math.floor(y):
    vect[0] = abs(y - math.floor(y)) * v1[int(x), math.ceil(y)] \
                 + abs(y - math.ceil(y)) * v1[int(x), math.floor(y)]
    vect[1] = abs(y - math.floor(y)) * v2[int(x), math.ceil(y)] \
                 + abs(y - math.ceil(y)) * v2[int(x), math.floor(y)]
  else:
    vect[0] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * v1[math.ceil(x), math.ceil(y)] \
           + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * v1[math.ceil(x), math.floor(y)] \
           + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * v1[math.floor(x), math.ceil(y)] \
           + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * v1[math.floor(x), math.floor(y)] 
    vect[1] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * v2[math.ceil(x), math.ceil(y)] \
           + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * v2[math.ceil(x), math.floor(y)] \
           + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * v2[math.floor(x), math.ceil(y)] \
           + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * v2[math.floor(x), math.floor(y)] 
  return (vect)

def batch_vect_interp_2d(x, y, vector_field):
  num_tens = x.shape[0]
  vect = np.zeros((num_tens, 2))
  v1 = vector_field[0]
  v2 = vector_field[1]

  for p in range(num_tens):
    if ((math.floor(x[p]) < 0) or (math.ceil(x[p]) >= v1.shape[0]) or (math.floor(y[p]) < 0) or (math.ceil(y[p]) >= v1.shape[1])):
      # data is out of bounds, return identity vector
      vect[p,0] = 1
      vect[p,1] = 1
      continue

    if x[p] == math.floor(x[p]) and y[p] == math.floor(y[p]):
      vect[p,0] = v1[int(x[p]), int(y[p])]
      vect[p,1] = v2[int(x[p]), int(y[p])]
    elif x[p] != math.floor(x[p]) and y[p] == math.floor(y[p]):
      vect[p,0] = abs(x[p] - math.floor(x[p])) * v1[math.ceil(x[p]), int(y[p])] \
                     + abs(x[p] - math.ceil(x[p])) * v1[math.floor(x[p]), int(y[p])]
      vect[p,1] = abs(x[p] - math.floor(x[p])) * v2[math.ceil(x[p]), int(y[p])] \
                     + abs(x[p] - math.ceil(x[p])) * v2[math.floor(x[p]), int(y[p])]
    elif x[p] == math.floor(x[p]) and y[p] != math.floor(y[p]):
      vect[p,0] = abs(y[p] - math.floor(y[p])) * v1[int(x[p]), math.ceil(y[p])] \
                     + abs(y[p] - math.ceil(y[p])) * v1[int(x[p]), math.floor(y[p])]
      vect[p,1] = abs(y[p] - math.floor(y[p])) * v2[int(x[p]), math.ceil(y[p])] \
                     + abs(y[p] - math.ceil(y[p])) * v2[int(x[p]), math.floor(y[p])]
    else:
      vect[p,0] = abs(x[p] - math.floor(x[p])) * abs(y[p] - math.floor(y[p])) * v1[math.ceil(x[p]), math.ceil(y[p])] \
               + abs(x[p] - math.floor(x[p])) * abs(y[p] - math.ceil(y[p])) * v1[math.ceil(x[p]), math.floor(y[p])] \
               + abs(x[p] - math.ceil(x[p])) * abs(y[p] - math.floor(y[p])) * v1[math.floor(x[p]), math.ceil(y[p])] \
               + abs(x[p] - math.ceil(x[p])) * abs(y[p] - math.ceil(y[p])) * v1[math.floor(x[p]), math.floor(y[p])] 
      vect[p,1] = abs(x[p] - math.floor(x[p])) * abs(y[p] - math.floor(y[p])) * v2[math.ceil(x[p]), math.ceil(y[p])] \
               + abs(x[p] - math.floor(x[p])) * abs(y[p] - math.ceil(y[p])) * v2[math.ceil(x[p]), math.floor(y[p])] \
               + abs(x[p] - math.ceil(x[p])) * abs(y[p] - math.floor(y[p])) * v2[math.floor(x[p]), math.ceil(y[p])] \
               + abs(x[p] - math.ceil(x[p])) * abs(y[p] - math.ceil(y[p])) * v2[math.floor(x[p]), math.floor(y[p])] 
  return (vect)

def tens_interp_2d(x, y, tensor_field):
  tens = np.zeros((2, 2))
  eps11 = tensor_field[...,0,0]
  eps12 = tensor_field[...,0,1]
  eps21 = tensor_field[...,1,0]
  eps22 = tensor_field[...,1,1]
  if (math.floor(x) < 0) or (math.ceil(x) >= eps11.shape[0]) or (math.floor(y) < 0) or (math.ceil(y) >= eps11.shape[1]):
     # data is out of bounds, return identity tensor
     tens[0,0] = 1
     tens[1,1] = 1
     return(tens)
  if x == math.floor(x) and y == math.floor(y):
    tens[0, 0] = eps11[int(x), int(y)]
    tens[0, 1] = eps12[int(x), int(y)]
    tens[1, 0] = eps21[int(x), int(y)]
    tens[1, 1] = eps22[int(x), int(y)]
  elif x == math.floor(x) and y != math.floor(y):
    tens[0, 0] = abs(y - math.floor(y)) * eps11[int(x), math.ceil(y)] + \
           abs(y - math.ceil(y)) * eps11[int(x), math.floor(y)]
    tens[0, 1] = abs(y - math.floor(y)) * eps12[int(x), math.ceil(y)] + \
           abs(y - math.ceil(y)) * eps12[int(x), math.floor(y)]
    tens[1, 0] = abs(y - math.floor(y)) * eps21[int(x), math.ceil(y)] + \
           abs(y - math.ceil(y)) * eps21[int(x), math.floor(y)]
    tens[1, 1] = abs(y - math.floor(y)) * eps22[int(x), math.ceil(y)] + \
           abs(y - math.ceil(y)) * eps22[int(x), math.floor(y)]
  elif x != math.floor(x) and y == math.floor(y):
    tens[0, 0] = abs(x - math.floor(x)) * eps11[math.ceil(x), int(y)] + \
           abs(x - math.ceil(x)) * eps11[math.floor(x), int(y)]
    tens[0, 1] = abs(x - math.floor(x)) * eps12[math.ceil(x), int(y)] + \
           abs(x - math.ceil(x)) * eps12[math.floor(x), int(y)]
    tens[1, 0] = abs(x - math.floor(x)) * eps21[math.ceil(x), int(y)] + \
           abs(x - math.ceil(x)) * eps21[math.floor(x), int(y)]
    tens[1, 1] = abs(x - math.floor(x)) * eps22[math.ceil(x), int(y)] + \
           abs(x - math.ceil(x)) * eps22[math.floor(x), int(y)]
  else:
    tens[0, 0] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * eps11[math.ceil(x), math.ceil(y)] \
           + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * eps11[math.ceil(x), math.floor(y)] \
           + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * eps11[math.floor(x), math.ceil(y)] \
           + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * eps11[math.floor(x), math.floor(y)]
    tens[0, 1] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * eps12[math.ceil(x), math.ceil(y)] \
           + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * eps12[math.ceil(x), math.floor(y)] \
           + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * eps12[math.floor(x), math.ceil(y)] \
           + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * eps12[math.floor(x), math.floor(y)]
    tens[1, 0] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * eps21[math.ceil(x), math.ceil(y)] \
           + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * eps21[math.ceil(x), math.floor(y)] \
           + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * eps21[math.floor(x), math.ceil(y)] \
           + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * eps21[math.floor(x), math.floor(y)]
    tens[1, 1] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * eps22[math.ceil(x), math.ceil(y)] \
           + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * eps22[math.ceil(x), math.floor(y)] \
           + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * eps22[math.floor(x), math.ceil(y)] \
           + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * eps22[math.floor(x), math.floor(y)]

  return (tens)

def batch_tens_interp_2d(x, y, tensor_mat):
      #print("Warning! reenable njit")
  num_tens = x.shape[0]
  tens = np.zeros((num_tens, 2, 2))
  eps11 = tensor_mat[...,0,0]
  eps12 = tensor_mat[...,0,1]
  eps21 = tensor_mat[...,1,0]
  eps22 = tensor_mat[...,1,1]

  x = np.where(x<np.zeros_like(x),np.zeros_like(x),x)
  x = np.where(x>=np.ones_like(x)*(eps11.shape[0]-1),np.ones_like(x)*(eps11.shape[0]-1),x)
  y = np.where(y<np.zeros_like(y),np.zeros_like(y),y)
  y = np.where(y>=np.ones_like(x)*(eps11.shape[1]-1),np.ones_like(x)*(eps11.shape[1]-1),y)

  ceil_x = np.ceil(x).astype(int)
  floor_x = np.floor(x).astype(int)
  ceil_y = np.ceil(y).astype(int)
  floor_y = np.floor(y).astype(int)
  x_minus_floor_x = np.abs(x - floor_x)
  x_minus_ceil_x = np.abs(x - ceil_x)
  y_minus_floor_y = np.abs(y - floor_y)
  y_minus_ceil_y = np.abs(y - ceil_y)
  
  for p in range(num_tens):
    if x[p] == floor_x[p] and y[p] == floor_y[p]:
      # Find index where no interpolation is needed and just copy the values
      tens[p,0,0] = eps11[floor_x[p],floor_y[p]]
      tens[p,0,1] = eps12[floor_x[p],floor_y[p]]
      tens[p,1,0] = tens[p,0,1]
      tens[p,1,1] = eps22[floor_x[p],floor_y[p]]
    elif x[p] != floor_x[p] and y[p] == floor_y[p]:      
      tens[p,0,0] = x_minus_floor_x[p] * eps11[ceil_x[p], y[p].astype(int)] \
           + x_minus_ceil_x[p] * eps11[floor_x[p], y[p].astype(int)] 
      tens[p,0,1] = x_minus_floor_x[p] * eps12[ceil_x[p], y[p].astype(int)] \
           + x_minus_ceil_x[p] * eps12[floor_x[p], y[p].astype(int)] 
      tens[p,1,0] = tens[p,0,1]
      tens[p,1,1] = x_minus_floor_x[p] * eps22[ceil_x[p], y[p].astype(int)] \
           + x_minus_ceil_x[p] * eps22[floor_x[p], y[p].astype(int)] 
    elif x[p] == floor_x[p] and y[p] != floor_y[p]:
      tens[p,0,0] = y_minus_floor_y[p] * eps11[x[p].astype(int), ceil_y[p]] \
           + y_minus_ceil_y[p] * eps11[x[p].astype(int), floor_y[p]]
      tens[p,0,1] = y_minus_floor_y[p] * eps12[x[p].astype(int), ceil_y[p]] \
           + y_minus_ceil_y[p] * eps12[x[p].astype(int), floor_y[p]]
      tens[p,1,0] = tens[p,0,1]
      tens[p,1,1] = y_minus_floor_y[p] * eps22[x[p].astype(int), ceil_y[p]] \
           + y_minus_ceil_y[p] * eps22[x[p].astype(int), floor_y[p]]
    else:
      floor_x_floor_y = x_minus_floor_x[p] * y_minus_floor_y[p]
      floor_x_ceil_y = x_minus_floor_x[p] * y_minus_ceil_y[p]
      ceil_x_floor_y = x_minus_ceil_x[p] * y_minus_floor_y[p]
      ceil_x_ceil_y = x_minus_ceil_x[p] * y_minus_ceil_y[p]
      tens[p,0,0] = floor_x_floor_y * eps11[ceil_x[p], ceil_y[p]] \
           + floor_x_ceil_y * eps11[ceil_x[p], floor_y[p]] \
           + ceil_x_floor_y * eps11[floor_x[p], ceil_y[p]] \
           + ceil_x_ceil_y * eps11[floor_x[p], floor_y[p]] 
      tens[p,0,1] = floor_x_floor_y * eps12[ceil_x[p], ceil_y[p]] \
           + floor_x_ceil_y * eps12[ceil_x[p], floor_y[p]] \
           + ceil_x_floor_y * eps12[floor_x[p], ceil_y[p]] \
           + ceil_x_ceil_y * eps12[floor_x[p], floor_y[p]] 
      tens[p,1,0] = tens[p,0,1]
      tens[p,1,1] = floor_x_floor_y * eps22[ceil_x[p], ceil_y[p]] \
           + floor_x_ceil_y * eps22[ceil_x[p], floor_y[p]] \
           + ceil_x_floor_y * eps22[floor_x[p], ceil_y[p]] \
           + ceil_x_ceil_y * eps22[floor_x[p], floor_y[p]] 
  
  return (tens)

# compute eigenvectors according to A Method for Fast Diagonalization of a 2x2 or 3x3 Real Symmetric Matrix
# M.J. Kronenburg
# https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwj6zeiLut3qAhUPac0KHcyjDn4QFjAGegQIAxAB&url=https%3A%2F%2Farxiv.org%2Fpdf%2F1306.6291&usg=AOvVaw0BbaDECw-ghHGxek-LaB33
def eigv(tens):
    # TODO check dimensions, for now assuming 2D
    # The hope is that this implementation fixes the sign issue
    phi = 0.5 * np.arctan2(2 * tens[:,:,0,1] , (tens[:,:,0,0] - tens[:,:,1,1]))
    vs = np.zeros_like(tens)
    # want vs [:,:,0] to contain the eigenvector corresponding to the smallest
    # eigenvalue, which is \lambda_2 from the link above ie [-sin(\phi) cos(\phi)]
    vs[:,:,1,0] = np.cos(phi)
    vs[:,:,1,1] = np.sin(phi)
    vs[:,:,0,1] = vs[:,:,1,0] # cos(phi)
    vs[:,:,0,0] = -vs[:,:,1,1] # -sin(phi)
    return (vs)

def eigv_up(tens):
    # Compute eigenvectors for 2D tensors stored in upper triangular format
    # TODO check dimensions, for now assuming 2D
    # The hope is that this implementation fixes the sign issue
    phi = 0.5 * np.arctan2(2 * tens[:,:,1] , (tens[:,:,0] - tens[:,:,2]))
    vs = np.zeros_like(tens)
    # want vs [:,:,0] to contain the eigenvector corresponding to the smallest
    # eigenvalue, which is \lambda_2 from the link above ie [-sin(\phi) cos(\phi)]
    vs[:,:,1,0] = np.cos(phi)
    vs[:,:,1,1] = np.sin(phi)
    vs[:,:,0,1] = vs[:,:,1,0] # cos(phi)
    vs[:,:,0,0] = -vs[:,:,1,1] # -sin(phi)
    return (vs)

def eigv_3d(tens):
  # Find principal eigenvectors of 3d tensor field.
  eigenvals, eigenvecs = np.linalg.eigh(tens)
  return (eigenvecs)

def get_framework(arr):
  # return np or torch depending on type of array
  # also returns framework name as "numpy" or "torch"
  fw = None
  fw_name = ''
  if type(arr) == np.ndarray:
    fw = np
    fw_name = 'numpy'
  else:
    fw = torch
    fw_name = 'torch'
  return (fw, fw_name)


def eigv_sign_deambig(eigenvecs):
  # deambiguate eigenvector sign in each direction independently
  # want center pixel eigenvector to have same sign as both neighbors, when both neighbors sign matches
  # lr_dot, bt_dot, rf_dot > 0 ==> neighbors sign matches
  # lp_dot, bp_dot, rp_dot < 0 ==> pixel sign does not match neighbor
  fw, fw_name = get_framework(eigenvecs)
  if fw_name == "numpy":
    vecsx = fw.copy(eigenvecs)
    vecsy = fw.copy(eigenvecs)
    vecsz = fw.copy(eigenvecs)
  else:
    vecsx = fw.clone(eigenvecs)
    vecsy = fw.clone(eigenvecs)
    vecsz = fw.clone(eigenvecs)

  #lp_dot = np.einsum('...j,...j',eigenvecs[:-1,:,:,:,2],eigenvecs[1:,:,:,:,2])
  #lr_dot = np.einsum('...j,...j',eigenvecs[:-2,:,:,:,2],eigenvecs[2:,:,:,:,2])
  #bp_dot = np.einsum('...j,...j',eigenvecs[:,:-1,:,:,2],eigenvecs[:,1:,:,:,2])
  #bt_dot = np.einsum('...j,...j',eigenvecs[:,:-2,:,:,2],eigenvecs[:,2:,:,:,2])
  #rp_dot = np.einsum('...j,...j',eigenvecs[:,:,:-1,:,2],eigenvecs[:,:,1:,:,2])
  #rf_dot = np.einsum('...j,...j',eigenvecs[:,:,:-2,:,2],eigenvecs[:,:,2:,:,2])
  lp_dot = fw.einsum('...j,...j',eigenvecs[:-1,:,:,:],eigenvecs[1:,:,:,:])
  lr_dot = fw.einsum('...j,...j',eigenvecs[:-2,:,:,:],eigenvecs[2:,:,:,:])
  bp_dot = fw.einsum('...j,...j',eigenvecs[:,:-1,:,:],eigenvecs[:,1:,:,:])
  bt_dot = fw.einsum('...j,...j',eigenvecs[:,:-2,:,:],eigenvecs[:,2:,:,:])
  rp_dot = fw.einsum('...j,...j',eigenvecs[:,:,:-1,:],eigenvecs[:,:,1:,:])
  rf_dot = fw.einsum('...j,...j',eigenvecs[:,:,:-2,:],eigenvecs[:,:,2:,:])
  for xx in range(eigenvecs.shape[0]):
    for yy in range(eigenvecs.shape[1]):
      for zz in range(eigenvecs.shape[2]):
        if xx < lr_dot.shape[0]:
          if lr_dot[xx,yy,zz] > 0 and lp_dot[xx,yy,zz] < 0:
            vecsx[xx+1,yy,zz,:] = -vecsx[xx+1,yy,zz,:]
        if yy < bt_dot.shape[1]:
          if bt_dot[xx,yy,zz] > 0 and bp_dot[xx,yy,zz] < 0:
            vecsy[xx,yy+1,zz,:] = -vecsy[xx,yy+1,zz,:]
        if zz < rf_dot.shape[2]:
          if rf_dot[xx,yy,zz] > 0 and rp_dot[xx,yy,zz] < 0:
            vecsz[xx,yy,zz+1,:] = -vecsz[xx,yy,zz+1,:]
            
  return (vecsx, vecsy, vecsz)

# def eigv_up_3d(tens):
#     # Compute eigenvectors for 3D tensors stored in upper triangular format
#     # TODO check dimensions, for now assuming 3D
#     # The hope is that this implementation fixes the sign issue
# NOT IMPLEMENTED YET
#     phi = 0.5 * np.arctan2(2 * tens[:,:,:,1] , (tens[:,:,:,0] - tens[:,:,:,2]))
#     vs = np.zeros_like(tens)
#     # want vs [:,:,0] to contain the eigenvector corresponding to the smallest
#     # eigenvalue, which is \lambda_2 from the link above ie [-sin(\phi) cos(\phi)]
#     vs[:,:,:,1,0] = np.cos(phi)
#     vs[:,:,:,1,1] = np.sin(phi)
#     vs[:,:,:,0,1] = vs[:,:,:,1,0] # cos(phi)
#     vs[:,:,:,0,0] = -vs[:,:,:,1,1] # -sin(phi)
#     return (vs)

def make_pos_def(tens, mask, small_eval = 0.00005):
  # make any small or negative eigenvalues slightly positive and then reconstruct tensors
  evals, evecs = np.linalg.eig(tens)
  #np.abs(evals, out=evals)
  idx = np.where(evals < small_eval)
  #idx = np.where(evals < 0)
  num_found = 0
  #print(len(idx[0]), 'tensors found with eigenvalues <', small_eval)
  for ee in range(len(idx[0])):
    if mask[idx[0][ee], idx[1][ee], idx[2][ee]]:
      num_found += 1
      evals[idx[0][ee], idx[1][ee], idx[2][ee], idx[3][ee]] = small_eval

  print(num_found, 'tensors found with eigenvalues <', small_eval)
  #print(num_found, 'tensors found with eigenvalues < 0')
  mod_tens = np.einsum('...ij,...jk,...k,...lk->...il',
                       evecs, np.identity(3), evals, evecs)
  return(mod_tens)
    
 
def scale_by_alpha(tensors, alpha):
  # This scaling function assumes that the input provided for scaling are diffusion tensors
  # and hence scales by 1/e^{\alpha}.
  # If the inverse-tensor metric is provided instead, we would need to scale by e^\alpha
  out_tensors = np.copy(tensors)

  if tensors.shape[2] == 3:
    for kk in range(3): 
      out_tensors[:,:,kk] /= np.exp(alpha)
  elif tensors.shape[2:] == (2, 2):
    for jj in range(2):
      for kk in range(2):
        out_tensors[:,:,jj,kk] /= np.exp(alpha)
  elif tensors.shape[3] == 6:
    for kk in range(6): 
      out_tensors[:,:,:,kk] /= np.exp(alpha)
  elif tensors.shape[3:] == (3, 3):
    for jj in range(3):
      for kk in range(3):
        out_tensors[:,:,:,jj,kk] /= np.exp(alpha)
  else:
    print(tensors.shape, "unexpected tensor shape")
  return(out_tensors)

def threshold_to_input(tens_to_thresh, input_tens, mask, ratio=1.0):
  # scale the tens_to_thresh by the ratio * norm^2 of the largest tensor in input_tens
  # assumes input tens are full 2x2 tensors
  # TODO confirm that ratio is between 0 and 1
  if input_tens.shape[2] == 3:
    norm_in_tens = np.linalg.norm(input_tens,axis=(2))
  elif input_tens.shape[2:] == (2,2):
    norm_in_tens = np.linalg.norm(input_tens,axis=(2,3))
  elif input_tens.shape[3] == 6:
    norm_in_tens = np.linalg.norm(input_tens,axis=(3))
  elif input_tens.shape[3:] == (3,3):
    norm_in_tens = np.linalg.norm(input_tens,axis=(3,4))
  else:
    print(input_tens.shape, "unexpected tensor shape")
  if tens_to_thresh.shape[2] == 3:
    norm_sq = np.linalg.norm(tens_to_thresh,axis=(2))
  elif tens_to_thresh.shape[2:] == (2,2):
    norm_sq = np.linalg.norm(tens_to_thresh,axis=(2,3))
  elif tens_to_thresh.shape[3] == 6:
    norm_sq = np.linalg.norm(tens_to_thresh,axis=(3))    
  elif tens_to_thresh.shape[3:] == (3,3):
    norm_sq = np.linalg.norm(tens_to_thresh,axis=(3,4))    
  else:
    print(tens_to_thresh.shape, "unexpected tensor shape")  
  norm_sq = norm_sq * norm_sq # norm squared of each tensor in tens_to_thresh

  # just square the threshold, no need to element-wise square the entire norm_in_tens matrix
  thresh = np.max(norm_in_tens)
  thresh = ratio * thresh * thresh
  
  thresh_tens = np.copy(tens_to_thresh)
  scale_factor = np.ones_like(norm_sq)
  scale_factor[norm_sq > thresh] = thresh / norm_sq[norm_sq > thresh]
  scale_factor[mask == 0] = 1

  if tens_to_thresh.shape[2] == 3:
    for kk in range(3): 
      thresh_tens[:,:,kk] *= scale_factor
  elif tens_to_thresh.shape[2:] == (2, 2):
    for jj in range(2):
      for kk in range(2):
        thresh_tens[:,:,jj,kk] *= scale_factor
  elif tens_to_thresh.shape[3] == 6:
    for kk in range(6): 
      thresh_tens[:,:,:,kk] *= scale_factor
  elif tens_to_thresh.shape[3:] == (3, 3):
    for jj in range(3):
      for kk in range(3):
        thresh_tens[:,:,:,jj,kk] *= scale_factor
  else:
    print(tens_to_thresh.shape, "unexpected tensor shape")

  return(thresh_tens)
# end threshold_to_input

  
