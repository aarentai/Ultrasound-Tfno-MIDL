# an app to run metricModSolver
# options include:
#   directory and file names to use
#   conditioning thresholds and other options
import os.path
from . import appTypes
from util import YAMLcfg
from data import io
from data import fileManager as fm
from algo.metricModSolver import solve_3d
from algo.metricModSolver2d import solve_2d

MetricEstimatorConfigSpec = {
  'paths': appTypes.CmdLineAppConfigSpec,
  'options': { 'dim':
               YAMLcfg.Param(default = 3,
                             comment = 'dimension of data, used to select between 2D and 3D versions of underlying code.  Valid values are either 2 or 3.'),
               'inputTensorSuffix':
               YAMLcfg.Param(default = 'input_tensors.nhdr',
                             comment = 'name of input tensor file, not including path'),
               'inputMaskSuffix':
               YAMLcfg.Param(default = 'input_mask.nhdr',
                             comment = 'name of input mask file, not including path'),
               'outputTensorSuffix':
               YAMLcfg.Param(default = 'output_tensors.nhdr',
                             comment = 'name of output tensor file scaled by the metric modulation factor alpha, not including path'),
               'min_eigenvalue':
                YAMLcfg.Param(default = 5e-3,
                              comment = 'Minimum eigenvalue allowed, all smaller eigenvalues will be set to this value'),
               'num_iterations':
               YAMLcfg.Param(default = 450,
                             comment = "Number of GMRES iterations to run"),
               'clipping_range':
               YAMLcfg.Param(default = [-2, 2],
                             comment = "Minimum and maximum allowed alpha values."),
               'sigma':
               YAMLcfg.Param(default = None,
                             comment = "Sigma for smoothing tensor field prior to estimation.  Setting to None will skip filtering"),
               'saveIntermediateResults':
               YAMLcfg.Param(default = False,
                             comment = "Save off results of intermediate calculations. Note: When True, the algorithm can consume a large amount of memory.")
             },
  '_resource': 'AdaptaMetric_metricEstimation'
}

def default_config():
  return YAMLcfg.SpecToConfig(MetricEstimatorConfigSpec)

def load_data(cfg):
  filename = f"{cfg.paths.inputPrefix}/{cfg.options.inputTensorSuffix}"
  froot, fext = os.path.splitext(filename)
  if fext == ".mat":
    data = io.loadDataFromMat(filename, cfg.options.matfileTensorElem)
    mask = io.loadDataFromMat(filename, cfg.options.matfileMaskElem)
  else:
    data = io.ReadTensors(filename)
    filename = f"{cfg.paths.inputPrefix}/{cfg.options.inputMaskSuffix}"
    mask = io.ReadScalars(filename)
  
  return data, mask


class MetricEstimator(appTypes.BasicApp):
  def __init__(self, config=None):
    cfg = config
    if config is None:
      cfg = default_config()
    super().__init__(solve_3d, cfg)
    self.name = "Metric Estimator"
    # TODO auto-detect 2D vs 3D
    # Note that 2D vs 3D also refers to whether the tensors themselves are 2D or 3D embedded in a 2D or 3D image.
    self.desc = """ Runs the metric estimator metricModSolver.
Set the config param options.dim to run the 2D metricModSolver2d"""

  def run(self):
    self.timer.reset()
    data, mask = load_data(self.cfg)
    is_2d = False
    if ((self.cfg.options.dim == 2) or
        (len(data.shape) == 3) or
        (len(data.shape) == 4 and (data.shape[0] == 2 or data.shape[3] == 2))):
      is_2d = True
      self.run_function = solve_2d
      
    fm.create_dir_hierarchy(self.cfg.paths.outputPrefix)
    self.cwd = self.cfg.paths.outputPrefix

    (alpha, out_tens, out_mask, rks, intermed_results) = self.run_function(
      data, mask, self.cfg.options.num_iterations, self.cfg.options.clipping_range, 1.0,
      self.cfg.options.saveIntermediateResults, self.cfg.options.min_eigenvalue,
      self.cfg.options.sigma)

    self.timer.pause()
    return(alpha, out_tens, out_mask, rks, intermed_results)
    
