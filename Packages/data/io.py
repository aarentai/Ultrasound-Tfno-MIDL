# data input/output helpers, includes conversion to/from various data formats
# We heavily rely on lazy module loading here, put lazy-loaded modules in __init__.py
# This way we don't load the module until it is actually used -- no sense loading SimpleITK or pytorch etc if a particular application does not use them.
import os.path
from lazy_imports import np, sitk, loadmat
from util import YAMLcfg
from . import nrrd
from .convert import GetSITKImageFromNP, GetNPArrayFromSITK

def writeYAMLConfig(spec, cfg, filename):
  cfstr = YAMLcfg.ConfigToYAML(spec, cfg)  
  with open(filename, "w") as f:
    f.write(cfstr)

def readYAMLConfig(spec, filename):
  yd = YAMLcfg.LoadYAMLDict(filename)
  return YAMLcfg.MkConfig(yd, spec)

def writePath(x_coords, y_coords, filename):
  with open(filename, 'wb') as f:
    np.save(f, x_coords)
    np.save(f, y_coords)

def writePath3D(x_coords, y_coords, z_coords, filename):
  with open(filename, 'wb') as f:
    np.save(f, x_coords)
    np.save(f, y_coords)
    np.save(f, z_coords)

def readPath(filename):
  with open(filename, 'rb') as f:
    x_coords = np.load(f)
    y_coords = np.load(f)
  return(x_coords, y_coords)

def readPath3D(filename):
  with open(filename, 'rb') as f:
    x_coords = np.load(f)
    y_coords = np.load(f)
    z_coords = np.load(f)
  return(x_coords, y_coords, z_coords)

def loadDataFromMat(filename, elem):
  mat = loadmat(filename)
  return mat[elem]

def readRaw(filename,dtype=np.float64):
  # read raw data in
  # TODO allow for other datatypes besides double
  return(np.fromfile(filename, dtype))

def ReadTensors(filename):
  return(GetNPArrayFromSITK(sitk.ReadImage(filename),True))

def ReadScalars(filename):
  return(GetNPArrayFromSITK(sitk.ReadImage(filename)))

def WriteScalarNPArray(npimg, filename):
  sitk.WriteImage(GetSITKImageFromNP(npimg), filename)

def WriteScalarSITKImage(sitkimg, filename):
  sitk.WriteImage(sitkimg, filename)
  
def WriteTensorNPArray(npimg, filename):
  # TODO Note that FixTensorHeader assumes that filename is a nrrd file. Fix this up to only do it when it is a nrrd header
  sitk.WriteImage(GetSITKImageFromNP(npimg, has_component_data=True), filename)
  is2D=True
  if len(npimg.shape) == 4:
    is2D=False
  fixTensorHeader(filename,is2D=is2D)

def Write2DTensorSITKImage(sitkimg, filename):
  # TODO Note that FixTensorHeader assumes that filename is a nrrd file. Fix this up to only do it when it is a nrrd header
  sitk.WriteImage(sitkimg, filename)
  is2D=True
  if len(sitkimg.GetSize()) == 4:
    is2D=False
  fixTensorHeader(filename,is2D=is2D)
  
def fixTensorHeader(filename,is2D=True):
  # sitk.WriteImage handles tensor components incorrectly.  Fix up several fields in the nrrd file.
  # In particular, this section of the nrrd header:
  #   space: left-posterior-superior
  #   sizes: 3 100 100
  #   space directions: (1,0,0) (0,1,0) (0,0,1)
  #   kinds: domain domain domain
  #   endian: little
  #   encoding: raw
  #   space origin: (0,0,0)
  # Should become for 2D tensor images:
  #   space dimension: 2
  #   sizes: 3 100 100
  #   space directions: none (1,0) (0,1)
  #   kinds: vector domain domain
  #   endian: little
  #   encoding: raw
  #   space origin: (0,0)
  # And this for 3D tensor images:
  #   space dimension: 3
  #   sizes: 3 100 100 100
  #   space directions: none (1,0,0) (0,1,0) (0,0,1)
  #   kinds: vector domain domain domain
  #   endian: little
  #   encoding: raw
  #   space origin: (0,0,0)
 
  tfm = nrrd.nrrdTransform()
  tfm.file = filename
  if is2D:
    tfm.to_2D_tensor()
  else:
    tfm.to_3D_tensor()
