from lazy_imports import np, sitk, torch

# Note that when SimpleITK converts to/from numpy, it also transposes the data
# So let's create our own routines that address that.  We want internal consistency here as best
# as possible.  This will have lots of copying expense, but since we expect to do most processing
# in numpy and/or pytorch, it's better to do the copying work for I/O and visualization

# Always call GetNPArrayFromSITK instead of sitk.GetArrayFromImage
# Always call GetNPArrayViewFromSITK instead of sitk.GetArrayViewFromImage
# Always call GetSITKImageFromNP instead of sitk.GetImageFromArray
# Always call io.WriteTensorNPArray instead of sitk.WriteImage for tensor np arrays
# Always call io.WriteTensorSITKImage instead of sitk.WriteImage for tensor images

def GetNPArrayFromSITK(sitkimg, has_component_data=False):
  # If RGB or tensor data etc, set has_component_data to True so that last dimension is not
  # transposed.
  # This assumes that the component data is in the last dimension.
  # TODO fix this assumption to work for component data in first dimension as well
  # Currently works for 2D and 3D images
  tmp_np = sitk.GetArrayFromImage(sitkimg)
  if has_component_data or (len(tmp_np.shape) != len(sitkimg.GetSize())):
    transpose_tuple=(1,0,2)
    if len(tmp_np.shape) == 4:
      transpose_tuple=(2,1,0,3)    
    return np.transpose(tmp_np,transpose_tuple)
  else:
    transpose_tuple=(1,0)
    if len(tmp_np.shape) == 3:
      transpose_tuple=(2,1,0)           
    return np.transpose(tmp_np, transpose_tuple)
  
def GetNPArrayViewFromSITK(sitkimg, has_component_data=False):
  # If RGB or tensor data etc, set has_component_data to True so that last dimension is not
  # transposed.
  # This assumes that the component data is in the last dimension.
  # TODO fix this assumption to work for component data in first dimension as well
  # Currently works for 2D and 3D images
  if has_component_data:
    transpose_tuple=(1,0,2)
    if len(sitkimg.GetSize()) == 4:
      transpose_tuple=(2,1,0,3)    
    return np.transpose(sitk.GetArrayViewFromImage(sitkimg),transpose_tuple)
  else:
    transpose_tuple=(1,0)
    if len(sitkimg.GetSize()) == 3:
      transpose_tuple=(2,1,0)
    elif len(sitkimg.GetSize()) == 4:
      transpose_tuple=(3,2,1,0)
    return np.transpose(sitk.GetArrayViewFromImage(sitkimg), transpose_tuple)

def GetSITKImageFromNP(npimg, has_component_data=False):
  # If RGB or tensor data etc, set has_component_data to True so that last dimension is not
  # transposed.
  # This assumes that the component data is in the last dimension.
  # TODO fix this assumption to work for component data in first dimension as well
  # Currently works for 2D and 3D images
  if has_component_data:
    transpose_tuple=(1,0,2)
    if len(npimg.shape) == 4:
      transpose_tuple=(2,1,0,3)    
    return sitk.GetImageFromArray(np.transpose(npimg,transpose_tuple))
  else:
    transpose_tuple=(1,0)
    if len(npimg.shape) == 3:
      transpose_tuple=(2,1,0)           
    return sitk.GetImageFromArray(np.transpose(npimg, transpose_tuple))

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

def read_nhdr(path, as_type='torch'):
    if as_type=='torch':
        return torch.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage(path)))
    else:
        return sitk.GetArrayFromImage(sitk.ReadImage(path))