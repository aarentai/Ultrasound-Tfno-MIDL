from lazy_imports import np, loadmat
from util import YAMLcfg
from disp.vis import vis_ellipses
from apps import geodesicShooting as gs

def testConfig():
  print("Testing geodesicShooting config")
  cfg = gs.default_config()
  cfstr = YAMLcfg.ConfigToYAML(gs.geodesicShootingConfigSpec, cfg)
  print("Default Config:\n",cfstr)
  return(cfg)

def updateConfig(config, outSuffix=None, matfileElem=None, do_transpose=None, transpose=None, delta_t=None, num_iters=None):
  # set options in config, if None revert to default
  default = gs.default_config()

  # TODO seems like there could be a more programmatic way to set these instead of all the repetitive code below.  Want something that is really short and easy to specify and update for multiple test cases.
  if outSuffix is None:
    config.options.outputPathSuffix = default.options.outputPathSuffix
  else:
    config.options.outputPathSuffix = outSuffix
    
  if matfileElem is None:
    config.options.matfileTensorElem = default.options.matfileTensorElem
  else:
    config.options.matfileTensorElem = matfileElem

  if do_transpose is None:
    config.options.doDataTranspose = default.options.doDataTranspose
  else:
    config.options.doDataTranspose = do_transpose

  if transpose is None:
    config.options.transpose = default.options.transpose
  else:
    config.options.transpose = transpose

  if delta_t is None:
    config.options.delta_t = default.options.delta_t
  else:
    config.options.delta_t = delta_t

  if num_iters is None:
    config.options.num_iterations = default.options.num_iterations
  else:
    config.options.num_iterations = num_iters

  return(config)

def testShooting(config):
  print("\n\nTesting geodesic shooting with config:\n", YAMLcfg.ConfigToYAML(gs.geodesicShootingConfigSpec, config))
  shooter = gs.GeodesicShooter(config)
  points_x, points_y = shooter.run()
  print(f"Took {shooter.get_run_time()} seconds for geodesic shooting")
  return (points_x, points_y)

def testEuler(config):
  print("\n\nTesting euler integration with config:\n", YAMLcfg.ConfigToYAML(gs.geodesicShootingConfigSpec, config))
  integrator = gs.EulerIntegrator(config)
  points_x, points_y = integrator.run()
  print(f"Took {integrator.get_run_time()} seconds for Euler integration")
  return (points_x, points_y)


if __name__ == "__main__":

  cfg = testConfig()

  # Set up config options that will be the same for all runs
  # This assumes we are calling this test from metpy directory
  # TODO make this path more robust to calling location
  cfg.paths.inputPrefix = 'test/input' 
  cfg.paths.outputPrefix = 'test/output'
  
  cfg.options.inputTensorSuffix = 'annulus_tensors.mat'
  cfg.options.start_coordinate = (18.0, 53.0)
  # set initial velocity to None to have the geodesic shooter use the
  # principal eigenvector of the tensor field at the start coordinate for
  # the initial velocity.
  cfg.options.initial_velocity = None

  geo_delta_t = 0.005
  geo_iters = 25000 # 18000
  euler_delta_t = 0.01
  euler_iters = 12000 # 9600
  
  # Set up config options for each test case
  cfg = updateConfig(cfg, 'original_annulus_geo_path.npy', 'orig', True, (2,0,1), geo_delta_t, geo_iters)
  (geodesic_points_x, geodesic_points_y) = testShooting(cfg)
  # we want to plot the original tensor field below, so use this config to specify which data to load.
  data, mask = gs.load_data(cfg)

  cfg = updateConfig(cfg, 'analytic_annulus_geo_path.npy', 'analytic', True, (2,0,1), geo_delta_t, geo_iters)
  (geodesic_points_analytic_x, geodesic_points_analytic_y) = testShooting(cfg)
    
  cfg = updateConfig(cfg, 'gmres_alpha_annulus_geo_path.npy', 'gmres_alpha', True, (2,0,1), geo_delta_t, geo_iters)
  (geodesic_points_solved_x, geodesic_points_solved_y) = testShooting(cfg)

  cfg = updateConfig(cfg, 'gmres_inv_annulus_geo_path.npy', 'gmres_inv', True, (2,0,1), geo_delta_t, geo_iters)
  (geodesic_points_inv_x, geodesic_points_inv_y) = testShooting(cfg)

  cfg = updateConfig(cfg, 'euc_inv_annulus_geo_path.npy', 'euc_inv', True, (2,0,1), geo_delta_t, geo_iters)
  (geodesic_points_solved_CG_x, geodesic_points_solved_CG_y) = testShooting(cfg)

  cfg = updateConfig(cfg, 'original_annulus_euler_path.npy', 'orig', True, (2,0,1), euler_delta_t, euler_iters)
  (eulerpath_points_x, eulerpath_points_y) = testEuler(cfg)

  vis_ellipses(data,
               'test geodesic shooting with annulus',
               geodesic_points_x, geodesic_points_y,
               geodesic_points_analytic_x, geodesic_points_analytic_y,
               geodesic_points_solved_CG_x, geodesic_points_solved_CG_y,
               geodesic_points_solved_x, geodesic_points_solved_y,
               geodesic_points_inv_x, geodesic_points_inv_y,
               eulerpath_points_x, eulerpath_points_y,
               True, f'{cfg.paths.outputPrefix}/shooting.png')

