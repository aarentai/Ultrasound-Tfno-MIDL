from lazy_imports import np
from lazy_imports import savemat
from data.io import ReadTensors, ReadScalars, WriteTensorNPArray, WriteScalarNPArray
from algo import euler, geodesic
import algo.metricModSolver as mms

if __name__ == "__main__":
  indir = '/usr/sci/projects/HCP/Kris/NSFCRCNS/TestData/3D/'
  outdir = '/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/working_3d_python/'
  #for_mat_file = {}
  cases = []
  all_start_coords = []
  cases.append('103818')
  all_start_coords.append([[62,126,56]])
  #cases.append('105923')
  #all_start_coords.append([[61,125,56]])
  #cases.append('111312')
  #all_start_coords.append([[62,128,56]])
  #cases.append('103818up2')
  #all_start_coords.append([[124,252,112]])
  #cases.append('105923up2')
  #all_start_coords.append([[122,250,112]])
  #cases.append('111312up2')
  #all_start_coords.append([[124,256,112]])
  
  for (cc, start_coords) in zip(cases, all_start_coords):
    run_case = f'{cc}'
    print("Running", run_case)
    for_mat_file = {}
    out_prefix = outdir + run_case

    subj = run_case[0:6]
    indir = '/usr/sci/projects/HCP/Kris/NSFCRCNS/prepped_data/' + subj + '/'
    if "up2" in run_case:
      #start_coords = [[122,250,112]]
      init_velocities = [None]
      tens_file = 'dti_1000_up2_tensor.nhdr'
      mask_file = 'dti_1000_up2_FA_mask.nhdr'
      t1_file = 't1_rescaled_orig_space_up2.nhdr'
      num_iters = 10000 # Way too much, but want to understand convergence for all brain cases
      save_intermediate_results = False
    else:
      #start_coords = [[61,125,56]]
      init_velocities = [None]
      tens_file = 'dti_1000_tensor.nhdr'
      mask_file = 'dti_1000_FA_mask.nhdr'
      t1_file = 't1_stripped_irescaled.nhdr'
      num_iters = 1000 # Way too much, but want to understand convergence for all brain cases
      save_intermediate_results = True
    # TODO Something funky is going on with thresh_ratio.  Seems better to skip thresholding altogether
    # Valid values are supposed to be between 0 and 1, with 1 meaning don't threshold anything.
    # And yet, we need a value of at least 2.0 to get decent results for cubics (or even better is to
    # skip the thresholding altogether).
    # TODO test thresholds with real data, and decide whether to skip altogether.
    thresh_ratio = 1.0
    tens_scale = 0.003

    in_tens = ReadTensors(indir+'/'+tens_file)
    in_mask = ReadScalars(indir+'/'+mask_file)
    if t1_file:
      in_T1 = ReadScalars(indir+'/'+t1_file)
    else:
      in_T1 = in_mask

    xsz=in_mask.shape[0]
    ysz=in_mask.shape[1]
    zsz=in_mask.shape[2]

    alpha, out_tens, out_mask, rks, intermed_results = mms.solve_3d(in_tens, in_mask, num_iters, [-2,2],
                                                               thresh_ratio, save_intermediate_results)

    out_tens_tri = np.zeros((xsz,ysz,zsz,6))
    out_tens_tri[:,:,:,0] = out_tens[:,:,:,0,0]
    out_tens_tri[:,:,:,1] = out_tens[:,:,:,0,1]
    out_tens_tri[:,:,:,2] = out_tens[:,:,:,0,2]
    out_tens_tri[:,:,:,3] = out_tens[:,:,:,1,1]
    out_tens_tri[:,:,:,4] = out_tens[:,:,:,1,2]
    out_tens_tri[:,:,:,5] = out_tens[:,:,:,2,2]

    if save_intermediate_results:
      scaled_tens_tri = np.zeros((xsz,ysz,zsz,6))
      scaled_tens_tri[:,:,:,0] = intermed_results['scaled_tensors'][:,:,:,0,0]
      scaled_tens_tri[:,:,:,1] = intermed_results['scaled_tensors'][:,:,:,0,1]
      scaled_tens_tri[:,:,:,2] = intermed_results['scaled_tensors'][:,:,:,0,2]
      scaled_tens_tri[:,:,:,3] = intermed_results['scaled_tensors'][:,:,:,1,1]
      scaled_tens_tri[:,:,:,4] = intermed_results['scaled_tensors'][:,:,:,1,2]
      scaled_tens_tri[:,:,:,5] = intermed_results['scaled_tensors'][:,:,:,2,2]
    
    WriteTensorNPArray(out_tens_tri, out_prefix + f'_thresh_{thresh_ratio}_tensors.nhdr')
    WriteTensorNPArray(in_tens, out_prefix + '_orig_tensors.nhdr')
    WriteScalarNPArray(out_mask, out_prefix + '_filt_mask.nhdr')
    WriteScalarNPArray(alpha, out_prefix + '_alpha.nhdr')
    if t1_file:
      WriteScalarNPArray(in_T1, out_prefix + '_T1.nhdr')
    if save_intermediate_results:
      WriteTensorNPArray(scaled_tens_tri, out_prefix + f'_scaled_tensors.nhdr')

    for_mat_file['orig_tensors'] = in_tens
    for_mat_file['thresh_tensors'] = out_tens
    for_mat_file['alpha'] = alpha
    for_mat_file['T1'] = in_T1
    for_mat_file['filt_mask'] = out_mask
    for_mat_file['rks'] = rks
    if save_intermediate_results:
      for_mat_file['scaled_tensors'] = intermed_results['scaled_tensors']
      

    tens_4_path = np.transpose(in_tens,(3,0,1,2))
    thresh_tens_4_path = np.transpose(out_tens_tri,(3,0,1,2))
    if save_intermediate_results:
      scaled_tens_4_path = np.transpose(scaled_tens_tri,(3,0,1,2))

    geo_delta_t = 0.1#0.01#0.005
    #geo_iters = 6000 # 22000 for Kris annulus(delta_t=0.005), 32000 for cubic (delta_t=0.005)
    geo_iters = 2000 # 22000 for Kris annulus(delta_t=0.005), 32000 for cubic (delta_t=0.005)
    euler_delta_t = 0.1
    euler_iters = 4460 # 14600
    geox, geoy, geoz = geodesic.geodesicpath_3d(tens_4_path, out_mask,\
                                                start_coords[0], init_velocities[0], \
                                                geo_delta_t, iter_num=geo_iters, both_directions=True)
    threshgeox, threshgeoy, threshgeoz = geodesic.geodesicpath_3d(thresh_tens_4_path, out_mask,\
                                                               start_coords[0], init_velocities[0], \
                                                               geo_delta_t, iter_num=geo_iters, both_directions=True)
    if save_intermediate_results:
      scalegeox, scalegeoy, scalegeoz = geodesic.geodesicpath_3d(scaled_tens_4_path, out_mask,\
                                                                 start_coords[0], init_velocities[0], \
                                                                 geo_delta_t, iter_num=geo_iters, both_directions=True)
    eulx, euly, eulz = euler.eulerpath_3d(tens_4_path, out_mask,\
                                          start_coords[0], init_velocities[0], euler_delta_t, iter_num=euler_iters, both_directions=True)

    for_mat_file['tens_4_path'] = tens_4_path
    for_mat_file['orig_path'] = (geox, geoy, geoz)
    for_mat_file['thresh_path'] = (threshgeox, threshgeoy, threshgeoz)
    for_mat_file['euler_path'] = (eulx, euly, eulz)
    if save_intermediate_results:
      for_mat_file['scaled_tens_4_path'] = scaled_tens_4_path
      for_mat_file['scaled_path'] = (scalegeox, scalegeoy, scalegeoz)
      
    savemat(out_prefix + '_results.mat',for_mat_file)

  # end for each case
  #savemat(outdir + 'brain_results.mat',for_mat_file)

