import os.path

from lazy_imports import np
from . import appTypes
from util import YAMLcfg
from data import io
from algo.geodesic import geodesicpath, geodesicpath_3d
from algo.euler import eulerpath, eulerpath_3d
from algo.dijkstra import shortpath

geodesicShootingConfigSpec = {
  'paths': appTypes.CmdLineAppConfigSpec,
  'options': { 'delta_t': 
               YAMLcfg.Param(default = 0.15,
                             comment = 'Size of time step, smaller gives better convergence.'),
               'num_iterations':
               YAMLcfg.Param(default = 18000,
                             comment = 'Number of time steps to take.'),
               'start_coordinate':
               YAMLcfg.Param(default = (15.0, 53.0),
                             comment = '(x,y) coordinates to shoot from.'),
               'end_coordinate':
               YAMLcfg.Param(default = (85.0, 50.0),
                             comment = '(x,y) coordinates to end at for shortest path.'),
               'initial_velocity':
               YAMLcfg.Param(default = None,
                             comment = 'initial velocity for shooting.  If None, it will use the value of the tensor field at the start coordinate.'),
               'both_directions':
               YAMLcfg.Param(default = False,
                             comment = 'If True, run shooting starting with both initial_velocity and -initial_velocity'),
               'inputTensorSuffix':
               YAMLcfg.Param(default = 'input_tensors.nhdr',
                             comment = 'name of input tensor file, not including path'),
               'inputMaskFile':
               YAMLcfg.Param(default = 'input_mask.nhdr',
                             comment = 'name of input mask file, including path'), # include path here because this could be in a different directory than the input tensor
               'outputPathSuffix':
               YAMLcfg.Param(default = '',
                             comment = 'name of file for writing the resulting path. Leave empty if path should not be written to file.'),
               'matfileTensorElem':
               YAMLcfg.Param(default = 'orig',
                             comment = 'name of element from matlab .mat file containing tensor field to extract'),
               'matfileMaskElem':
               YAMLcfg.Param(default = 'mask',
                             comment = 'name of element from matlab .mat file containing mask image to extract'),
               'doDataTranspose':
               YAMLcfg.Param(default = False,
                             comment = 'Set to True if need to do a data transpose before running the geodesic shooting algorithm'),
               'transpose':
               YAMLcfg.Param(default = (0,1,2,3),
                             comment = 'The argument to numpy.transpose if doDataTranspose is True.  A common value to use is (3,0,1,2) for 3d, (2,0,1) for 2d.  The default value will result in no transpose.'),
  },
  '_resource': 'AdaptaMetric_geodesicShooting'
}

def default_config():
  return YAMLcfg.SpecToConfig(geodesicShootingConfigSpec)

def load_data(cfg):
  filename = f"{cfg.paths.inputPrefix}/{cfg.options.inputTensorSuffix}"
  froot, fext = os.path.splitext(filename)
  if fext == ".mat":
    data = io.loadDataFromMat(filename, cfg.options.matfileTensorElem)
    mask = io.loadDataFromMat(filename, cfg.options.matfileMaskElem)
  else:
    data = io.ReadTensors(filename)
    filename = f"{cfg.options.inputMaskFile}"
    mask = io.ReadScalars(filename)
  
  if cfg.options.doDataTranspose:
    data = np.transpose(data, cfg.options.transpose)
  
  return data, mask

class GeodesicShooter(appTypes.BasicApp):
  def __init__(self, config=None):
    cfg = config
    if config is None:
      cfg = default_config()
    super().__init__(geodesicpath_3d, cfg)

    self.name = "3D Geodesic Shooter"
    self.desc = """ Runs geodesic shooting from the given start point with the given initial velocity.  Returns an x,y,z path"""

  def run(self):
    # TODO could use args instead and then use the BasicApp run w/ arbitrary args instead
    # of implementing new run method here.  Challenge then is what to do with output handling
    self.timer.reset()

    data, mask = load_data(self.cfg)
      
    (x,y,z) = self.run_function(data, mask, self.cfg.options.start_coordinate, self.cfg.options.initial_velocity, self.cfg.options.delta_t, metric='withoutscaling', iter_num = self.cfg.options.num_iterations, filename = f"{self.cfg.paths.outputPrefix}/{self.cfg.options.outputPathSuffix}", both_directions = self.cfg.options.both_directions)

    self.timer.pause()
    return(x,y,z)

class EulerIntegrator(appTypes.BasicApp):
  def __init__(self, config=None):
    cfg = config
    if config is None:
      cfg = default_config()
    super().__init__(eulerpath_3d, cfg)

    self.name = "3D Euler Integrator"
    self.desc = """ Runs Euler integration from the given start point.  Returns an x,y,z path"""

  def run(self):
    # TODO could use args instead and then use the BasicApp run w/ arbitrary args instead
    # of implementing new run method here.  Challenge then is what to do with output handling
    self.timer.reset()

    data, mask = load_data(self.cfg)
      
    (x,y,z) = self.run_function(data, mask, self.cfg.options.start_coordinate, self.cfg.options.initial_velocity, self.cfg.options.delta_t, metric='withoutscaling', iter_num = self.cfg.options.num_iterations, filename = f"{self.cfg.paths.outputPrefix}/{self.cfg.options.outputPathSuffix}", both_directions = self.cfg.options.both_directions)

    self.timer.pause()
    return(x,y,z)

class GeodesicShooter2d(appTypes.BasicApp):
  def __init__(self, config=None):
    cfg = config
    if config is None:
      cfg = default_config()
    super().__init__(geodesicpath, cfg)

    self.name = "2D Geodesic Shooter"
    self.desc = """ Runs geodesic shooting from the given start point with the given initial velocity.  Returns an x,y path"""

  def run(self):
    # TODO could use args instead and then use the BasicApp run w/ arbitrary args instead
    # of implementing new run method here.  Challenge then is what to do with output handling
    self.timer.reset()

    data, mask = load_data(self.cfg)
      
    (x,y) = self.run_function(data, mask, self.cfg.options.start_coordinate, self.cfg.options.initial_velocity, self.cfg.options.delta_t, metric='withoutscaling', iter_num = self.cfg.options.num_iterations, filename = f"{self.cfg.paths.outputPrefix}/{self.cfg.options.outputPathSuffix}")

    self.timer.pause()
    return(x,y)

class EulerIntegrator2d(appTypes.BasicApp):
  def __init__(self, config=None):
    cfg = config
    if config is None:
      cfg = default_config()
    super().__init__(eulerpath, cfg)

    self.name = "2D Euler Integrator"
    self.desc = """ Runs euler integration from the given start point.  Returns an x,y path"""

  def run(self):
    # TODO could use args instead and then use the BasicApp run w/ arbitrary args instead
    # of implementing new run method here.  Challenge then is what to do with output handling
    self.timer.reset()

    data, mask = load_data(self.cfg)
      
    (x,y) = self.run_function(data, mask, self.cfg.options.start_coordinate, self.cfg.options.initial_velocity, self.cfg.options.delta_t, metric='withoutscaling', iter_num = self.cfg.options.num_iterations, filename = f"{self.cfg.paths.outputPrefix}/{self.cfg.options.outputPathSuffix}")

    self.timer.pause()
    return(x,y)

class ShortestPath2d(appTypes.BasicApp):
  def __init__(self, config=None):
    cfg = config
    if config is None:
      cfg = default_config()
    super().__init__(shortpath, cfg)

    self.name = "2D Shortest Path"
    self.desc = """ Runs dijkstra's shortest path algorithm from the given start point to the given end point.  Returns an x,y path and the distance"""

  def run(self):
    # TODO could use args instead and then use the BasicApp run w/ arbitrary args instead
    # of implementing new run method here.  Challenge then is what to do with output handling
    self.timer.reset()

    data, mask = load_data(self.cfg)
      
    (x,y,dist) = self.run_function(data, mask, self.cfg.options.start_coordinate, self.cfg.options.end_coordinate,
                                   filename = f"{self.cfg.paths.outputPrefix}/{self.cfg.options.outputPathSuffix}")

    self.timer.pause()
    return(x,y,dist)
