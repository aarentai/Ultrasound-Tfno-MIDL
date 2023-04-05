# Simulation to run metric estimation, followed by geodesic shooting for verification
from apps import metricEstimation as me
from apps import geodesicShooting as gs
from util import YAMLcfg
from disp.vis import vis_tensors, vis_path, disp_scalar_to_file
from disp.vis import  disp_vector_to_file, disp_tensor_to_file
from disp.vis import disp_gradG_to_file, disp_gradA_to_file
from lazy_imports import plt, np
from lazy_imports import savemat
from util.parsers import parseMetricEstStdout
import pickle
from data.io import WriteTensorNPArray, WriteScalarNPArray

# for each parameter set
#   run metric estimation
#   extract number of iterations and final energy from output, save and plot it
#   run geodesic shooting on results
# save off plots

# Note that having to rerun multiple times for multiple iteration settings is done because the metric estimation solver doesn't write out these results after every iteration.  If it did, we could just run once and inspect results.  Instead we have to run multiple times and aggregate results.

#TODO better to call these experiments?

def run_simulation(sim_name, input_tensor, input_mask, input_dir, output_root_dir, min_eigenvalues, num_iters, sigmas, start_coords, init_velocities):
  me_cfg = me.default_config()
  gs_cfg = gs.default_config()

  me_cfg.paths.inputPrefix = input_dir
  me_cfg.options.inputTensorSuffix = input_tensor
  me_cfg.options.inputMaskSuffix = input_mask
  me_cfg.options.outputTensorSuffix = 'scaled_tensors.nhdr'
  me_cfg.options.dim = 3
  # don't save intermediate results when running many cases, takes too much memory
  me_cfg.options.saveIntermediateResults = False
  
  gs_cfg.options.inputTensorSuffix = me_cfg.options.outputTensorSuffix
  gs_cfg.options.doDataTranspose = True
  gs_cfg.options.transpose = (3, 0, 1, 2)
  gs_cfg.options.both_directions = True
  
  # TODO do we want to pass these in instead?
  geo_delta_t = 0.1
  geo_iters = 3000 
  euler_delta_t = 0.1
  euler_iters = 4600

  results = []

  for min_eval in min_eigenvalues:
    for num_iter in num_iters:
      for sigma in sigmas:
        test_case = f"mineval_{min_eval}_n_{num_iter}_s_{sigma}"
        test_dir = f"{output_root_dir}/{test_case}/"
        me_cfg.paths.outputPrefix = test_dir
        me_cfg.options.min_eigenvalue = min_eval
        me_cfg.options.num_iterations = num_iter
        me_cfg.options.sigma = sigma
        
        result = {}
        result["test_case"] = test_case
        result["dir"] = test_dir
        result["tensor_file"] = f"{test_dir}/{me_cfg.options.outputTensorSuffix}"
        result["metricEst"] = {}
        result["metricEst"]["config"] = me_cfg
        
        met_est = me.MetricEstimator(me_cfg)
        try:
          print(f"Running metric estimation for {test_case}")
          (alpha, out_tens, out_mask, rks, intermed_results) = met_est.run()
        
          # saving off config, commands, time to run is good provenance tracking
          # TODO figure out how to automate/standardize this to make it easier
          #    for other sim writing later
          result["metricEst"]["alpha"] = alpha
          result["metricEst"]["time"] = met_est.get_run_time()
          result["metricEst"]["out_tens"] = out_tens
          result["metricEst"]["out_mask"] = out_mask
          result["metricEst"]["rks"] = rks
          #result["metricEst"]["intermed_results"] = intermed_results
        
          xsz = out_mask.shape[0]
          ysz = out_mask.shape[1]
          zsz = out_mask.shape[2]
          out_tens_tri = np.zeros((xsz,ysz,zsz,6))
          out_tens_tri[:,:,:,0] = out_tens[:,:,:,0,0]
          out_tens_tri[:,:,:,1] = out_tens[:,:,:,0,1]
          out_tens_tri[:,:,:,2] = out_tens[:,:,:,0,2]
          out_tens_tri[:,:,:,3] = out_tens[:,:,:,1,1]
          out_tens_tri[:,:,:,4] = out_tens[:,:,:,1,2]
          out_tens_tri[:,:,:,5] = out_tens[:,:,:,2,2]
        
          WriteTensorNPArray(out_tens_tri, f"{test_dir}/{me_cfg.options.outputTensorSuffix}")
          #WriteTensorNPArray(in_tens, f'{test_dir}/orig_tensors.nhdr')
          WriteScalarNPArray(out_mask, f'{test_dir}/filt_mask.nhdr')
          WriteScalarNPArray(alpha, f'{test_dir}/alpha.nhdr')
          success = True
          
        except Exception as e:
          print(f"Error {e} while running {test_case} with config {YAMLcfg.ConfigToYAML(me.MetricEstimatorConfigSpec, me_cfg)}")
          success = False
          
        result["metricEst"]["status"] = success
        
        #TODO log commands
        #TODO parse results
        gs_cfg.paths.outputPrefix = me_cfg.paths.outputPrefix
        gs_cfg.paths.inputPrefix = me_cfg.paths.outputPrefix
        gs_cfg.options.inputMaskFile = f'{test_dir}/filt_mask.nhdr'
        
        coord_results = []
        for coords in start_coords:
          for init_velocity in init_velocities:
            coord_res = {}
            coord_res["coords"] = coords
            coord_res["init_velocity"] = init_velocity
            coord_res["shooting"] = {}
            coord_res["euler"] = {}
            gs_cfg.options.start_coordinate = coords
            gs_cfg.options.initial_velocity = init_velocity
            gs_cfg.options.outputPathSuffix = f'{sim_name}_{coords}_{init_velocity}_geo_path.npy'
            gs_cfg.options.delta_t = geo_delta_t
            gs_cfg.options.num_iterations = geo_iters
        
            # TODO consider whether we want to construct new class instance for each test case
            # or reuse same existing class
            coord_res["shooting"]["config"] = gs_cfg
            shooter = gs.GeodesicShooter(gs_cfg)
            try:
              print(f"Running geodesic shooting for {test_case} and starting point {coords}")
              points_x, points_y, points_z = shooter.run()
              success = True
            except Exception as e:
              print(f"Error {e} while geodesic shooting for {test_case} with config {YAMLcfg.ConfigToYAML(gs.geodesicShootingConfigSpec, gs_cfg)}")
              success = False
        
            coord_res["shooting"]["status"] = success
            if success:
              coord_res["shooting"]["time"] = shooter.get_run_time()
              #TODO is it ok to save off all these x, y, z coords?  Not too expensive?
              coord_res["shooting"]["x"] = points_x
              coord_res["shooting"]["y"] = points_y
              coord_res["shooting"]["z"] = points_z
          
            gs_cfg.options.outputPathSuffix = f'{sim_name}_{coords}_{init_velocity}_euler_path.npy'
            gs_cfg.options.delta_t = euler_delta_t
            gs_cfg.options.num_iterations = euler_iters
        
            coord_res["euler"] = {}
            coord_res["euler"]["config"] = gs_cfg
            integrator = gs.EulerIntegrator(gs_cfg)
            try:
              print(f"Running Euler integration for {test_case} and starting point {coords}")
              points_x, points_y, points_z = integrator.run()
              success = True
            except Exception as e:
              print(f"Error {e} during Euler integration for {test_case} with config {YAMLcfg.ConfigToYAML(gs.geodesicShootingConfigSpec, gs_cfg)}")
              success = False
        
            coord_res["euler"]["status"] = success
            if success:
              coord_res["euler"]["time"] = integrator.get_run_time()
        
              #TODO is it ok to save off all these x, y, z coords?  Not too expensive?
              coord_res["euler"]["x"] = points_x
              coord_res["euler"]["y"] = points_y
              coord_res["euler"]["z"] = points_z
        
            coord_results.append(coord_res)
          # end for each initial velocity
        # end for each start coordinate
        result["paths"] = coord_results
        with open(f'{test_dir}/results.pkl', 'wb') as f:
          pickle.dump(result, f)
        #savemat(f"{test_dir}results.mat", result)
        if "up2" not in sim_name:
          # upsampled case takes too much space so don't save off in that case
          results.append(result)
      # end for each sigma
    # end for each num_iter
  # end for each max_eval
        
  print(f"Finished metricEstSim {sim_name}")
  return(results)

def summarize_results(results, sim_name, output_root_dir, subplot_nrows, subplot_ncols,
                      show_energy_plot):
  print("Not implemented yet for new version of metric estimation, fix up before running")
  return()
  num_test_cases = len(results)
  cases_per_fig = subplot_nrows * subplot_ncols
  new_fig = True
  tens_fig = None
  energy_fig = None
  fig_count = 1
  #fig_file_root = f"{sim_name}_paths_{fig_count}.png"
  # TODO decide whether want to keep using inputTensorSuffix from first result as background
  # tensor field to plot
  show_euler = True
  #colors = ['k', 'r', 'g', 'purple', 'b', 'orange']
  colors = ['k', 'tab:blue', 'tab:orange', 'tab:purple', 'tab:green', 'tab:red', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
  coloridx = 1
  case_count = 1

  data_cfg = gs.default_config()
  gs_cfg = results[0]["paths"][0]["shooting"]["config"]
  me_cfg = results[0]["metricEst"]["config"]
  data_cfg.paths.inputPrefix = me_cfg.paths.inputPrefix
  data_cfg.options.inputTensorSuffix = me_cfg.options.inputTensorSuffix
  data_cfg.options.inputMaskFile = gs_cfg.options.inputMaskFile
  data_cfg.options.doDataTranspose = gs_cfg.options.doDataTranspose
  data_cfg.options.transpose = gs_cfg.options.transpose
  
  tensors, mask = gs.load_data(data_cfg)

  scalar_files = ['divergence_rhs', 'rhs', 'output_result_rhs', 'output_alpha_final', 'output_result_lhs', 'usermult2_lhs']
  vector_files = ['cross_x_first_nablaTT', 'cross_x_second_nablaTT', 'cross_x_nablaTT',
                  'cross_y_first_nablaTT', 'cross_y_second_nablaTT', 'cross_y_nablaTT',
                  'cross_x_T','cross_y_T', 'output_dot_T',
                  'dot_T_after_cross_x_T', 'dot_T_after_cross_y_T',
                  'cross_x_sqrt_det_nablaTT', 'cross_x_sqrt_det_nablaTT_after_expend',
                  'cross_y_sqrt_det_nablaTT', 'cross_y_sqrt_det_nablaTT_after_expend']
  tensor_files = ['output_dot_g', 'output_dot_g_inv']
  gradG_files = ['cross_x_gradG', 'cross_y_gradG']
  gradA_files = ['cross_x_gradA', 'cross_y_gradA']
  
  for test in results:
    if test["metricEst"]["status"] == False:
      continue

    if new_fig:
      tens_fig = vis_tensors(tensors, f"{sim_name}", False)
      energy_fig = plt.figure()
      plt.title(f"{sim_name} energy")
      plt.xlabel('Iteration')
      plt.ylabel('Energy')
      energy_resid_fig = plt.figure()
      plt.title(f"{sim_name} residual norm vs energy")
      plt.xlabel('-log(Residual Norm)')
      plt.ylabel('Energy')
      resid_fig = plt.figure()
      plt.title(f"{sim_name} residual norm")
      plt.xlabel('Iteration')
      plt.ylabel('Residual Norm')


    label =  f"{test['test_case']}"

    # plot energies and residual
    if show_energy_plot:
      energies = parseMetricEstStdout(test["metricEst"]["stdout"][-1])
      vis_path(energies['KSP Residual norm']['iter'], energies['KSP Residual norm']['value'], resid_fig, label, colors[coloridx], 5, 1, False, yscale='log')
      vis_path(energies['current energy']['iter'], energies['current energy']['value'], energy_fig, label, colors[coloridx], 5, 1, False, yscale='log')
      vis_path(-np.log(energies['KSP Residual norm']['value']), energies['current energy']['value'], energy_resid_fig, label, colors[coloridx], 5, 1, False, yscale='log')
    
    # plot paths  
    # TODO still fit everything on one plot?  
    for path in test["paths"]:
      coords = path["coords"]
      if path["shooting"]["status"]:
        vis_path(path["shooting"]["x"], path["shooting"]["y"], tens_fig, label+"(Geodesic)", colors[coloridx], 20, 1, False)

      if show_euler and path["euler"]["status"]:
        vis_path(path["euler"]["x"], path["euler"]["y"], tens_fig, label+"(Euler)", colors[0], 20, 1, False)

    # plot images
    for sf in scalar_files:
      fname = f"{output_root_dir}/{label}/{sf}"
      try:
        disp_scalar_to_file(fname, fname + f'_{label}.png', label)
      except:
        pass
    for vf in vector_files:
      fname = f"{output_root_dir}/{label}/{vf}"
      outfiles = [fname + '_{}_{}.png'.format(label, i) for i in range(2)]
      try:
        disp_vector_to_file(fname, outfiles, vf)
      except:
        pass
    for tf in tensor_files:
      fname = f"{output_root_dir}/{label}/{tf}"
      outfiles = [fname + '_{}_{}.png'.format(label, i) for i in range(3)]
      try:
        disp_tensor_to_file(fname, outfiles, tf)
      except:
        pass
    for gf in gradG_files:
      fname = f"{output_root_dir}/{label}/{gf}"
      outfiles = [fname + '_{}_{}.png'.format(label, i) for i in range(6)]
      try:
        disp_gradG_to_file(fname, outfiles, gf)
      except:
        pass
    for af in gradA_files:
      fname = f"{output_root_dir}/{label}/{af}"
      outfiles = [fname + '_{}_{}.png'.format(label, i) for i in range(4)]
      try:
        disp_gradA_to_file(fname, outfiles, af) 
      except:
        pass
        
    # After first data set no more Euler   
    show_euler = False
    coloridx += 1
    if coloridx == len(colors):
      coloridx = 1
    new_fig = False
    case_count += 1
    if case_count > num_test_cases / 2.0:
      new_fig = True
      case_count = 1
      tens_fig.savefig(f"{output_root_dir}/{sim_name}_paths_{fig_count}.png")
      plt.close(tens_fig)
      resid_fig.savefig(f"{output_root_dir}/{sim_name}_residual_{fig_count}.png")
      plt.close(resid_fig)
      energy_fig.savefig(f"{output_root_dir}/{sim_name}_energy_{fig_count}.png")
      plt.close(energy_fig)
      energy_resid_fig.savefig(f"{output_root_dir}/{sim_name}_residual_vs_energy_{fig_count}.png")
      plt.close(energy_resid_fig)
      fig_count += 1
      show_euler = True


               

  
