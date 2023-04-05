from lazy_imports import sitk, np, torch

# TODO add package name to the naming convention here, once identify a good package name!!
IM_TYPE_PY = 0
IM_TYPE_SITK = 1
IM_TYPE_NP = 2
IM_TYPE_TORCH = 3

DATA_TYPE_UINT = 0
DATA_TYPE_INT = 1
DATA_TYPE_FLOAT = 2
DATA_TYPE_DOUBLE = 3

data_type_map = {DATA_TYPE_UINT : {"py": int},
                 DATA_TYPE_INT : {"py": int},
                 DATA_TYPE_FLOAT : {"py": float},
                 DATA_TYPE_DOUBLE: {"py": float}}

def fill_data_type_map(im_type):
  # don't call this until necessary because we only want to access the particular module
  # when needed.
  if im_type == IM_TYPE_SITK:
    if not data_type_map[DATA_TYPE_UINT].has_key("sitk"):
      #import SimpleITK as sitk
      data_type_map[DATA_TYPE_UINT]["sitk"] = sitk.sitkUInt8
      data_type_map[DATA_TYPE_INT]["sitk"] = sitk.sitkInt32
      data_type_map[DATA_TYPE_FLOAT]["sitk"] = sitk.sitkFloat32
      data_type_map[DATA_TYPE_DOUBLE]["sitk"] = sitk.sitkFloat64
  elif im_type == IM_TYPE_NP:
    if not data_type_map[DATA_TYPE_UINT].has_key("np"):
      #import numpy as np
      data_type_map[DATA_TYPE_UINT]["np"] = np.uint8
      data_type_map[DATA_TYPE_INT]["np"] = np.int32
      data_type_map[DATA_TYPE_FLOAT]["np"] = np.float32
      data_type_map[DATA_TYPE_DOUBLE]["np"] = np.float64
  elif im_type == IM_TYPE_TORCH:
    if not data_type_map[DATA_TYPE_UINT].has_key("torch"):
      #import torch
      data_type_map[DATA_TYPE_UINT]["torch"] = torch.uint8
      data_type_map[DATA_TYPE_INT]["torch"] = torch.int32
      data_type_map[DATA_TYPE_FLOAT]["torch"] = torch.float32
      data_type_map[DATA_TYPE_DOUBLE]["torch"] = torch.float64
      
