from lazy_imports import np
from lazy_imports import savemat
from data.io import ReadTensors, ReadScalars, WriteTensorNPArray, WriteScalarNPArray
from algo import euler, geodesic
import algo.metricModSolver as mms

if __name__ == "__main__":
  indir = '/usr/sci/projects/HCP/Kris/NSFCRCNS/TestData/3D/'
  outdir = '/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/working_3d_python/cubic_4000iter/'
  for_mat_file = {}
  
  for cc in range(1,8):
  #for cc in range(1,3):
    run_case = f'cubic{cc}'
    print("Running", run_case)
    for_mat_file[run_case] = {}
    out_prefix = outdir + run_case
    tens_file = f'metpy_3D_{run_case}_tens.nhdr'
    mask_file = f'metpy_3D_{run_case}_mask.nhdr'
    t1_file = None
    start_coords = [[13, 14, 20]] # use for newest gen version
    init_velocities = [None] #[[-0.44845807, -0.89380387]] # use for newest gen version
    num_iters = 4000 # A little much, but want to guarantee convergence for all cubic cases
    #num_iters = 200
    sigma = 1.5
    save_intermediate_results = True
    # TODO Something funky is going on with thresh_ratio.  Seems better to skip thresholding altogether
    # Valid values are supposed to be between 0 and 1, with 1 meaning don't threshold anything.
    # And yet, we need a value of at least 2.0 to get decent results for cubics (or even better is to
    # skip the thresholding altogether).
    # TODO test thresholds with real data, and decide whether to skip altogether.
    thresh_ratio = 1.0
    tens_scale = 0.1
    xlim=(40,95)
    ylim=(40,80)

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
                                                                    thresh_ratio, save_intermediate_results, sigma=sigma)

    out_tens_tri = np.zeros((xsz,ysz,zsz,6))
    out_tens_tri[:,:,:,0] = out_tens[:,:,:,0,0]
    out_tens_tri[:,:,:,1] = out_tens[:,:,:,0,1]
    out_tens_tri[:,:,:,2] = out_tens[:,:,:,0,2]
    out_tens_tri[:,:,:,3] = out_tens[:,:,:,1,1]
    out_tens_tri[:,:,:,4] = out_tens[:,:,:,1,2]
    out_tens_tri[:,:,:,5] = out_tens[:,:,:,2,2]

    # scaled_tens_tri = np.zeros((xsz,ysz,zsz,6))
    # scaled_tens_tri[:,:,:,0] = intermed_results['scaled_tensors'][:,:,:,0,0]
    # scaled_tens_tri[:,:,:,1] = intermed_results['scaled_tensors'][:,:,:,0,1]
    # scaled_tens_tri[:,:,:,2] = intermed_results['scaled_tensors'][:,:,:,0,2]
    # scaled_tens_tri[:,:,:,3] = intermed_results['scaled_tensors'][:,:,:,1,1]
    # scaled_tens_tri[:,:,:,4] = intermed_results['scaled_tensors'][:,:,:,1,2]
    # scaled_tens_tri[:,:,:,5] = intermed_results['scaled_tensors'][:,:,:,2,2]
    
    #WriteTensorNPArray(out_tens_tri, out_prefix + f'_thresh_{thresh_ratio}_tensors.nhdr')
    #WriteTensorNPArray(scaled_tens_tri, out_prefix + f'_scaled_tensors.nhdr')
    WriteTensorNPArray(out_tens_tri, out_prefix + f'_scaled_tensors.nhdr')
    WriteTensorNPArray(in_tens, out_prefix + '_orig_tensors.nhdr')
    WriteScalarNPArray(out_mask, out_prefix + '_filt_mask.nhdr')
    WriteScalarNPArray(alpha, out_prefix + '_alpha.nhdr')
    if t1_file:
      WriteScalarNPArray(in_T1, out_prefix + '_T1.nhdr')

    for_mat_file[run_case]['orig_tensors'] = in_tens
    #for_mat_file[run_case]['thresh_tensors'] = out_tens
    #for_mat_file[run_case]['scaled_tensors'] = intermed_results['scaled_tensors']
    for_mat_file[run_case]['scaled_tensors'] = out_tens
    for_mat_file[run_case]['filt_mask'] = out_mask
    for_mat_file[run_case]['alpha'] = alpha
    for_mat_file[run_case]['T1'] = in_T1
    for_mat_file[run_case]['rks'] = rks

    tens_4_path = np.transpose(in_tens,(3,0,1,2))
    #thresh_tens_4_path = np.transpose(out_tens_tri,(3,0,1,2))
    #scaled_tens_4_path = np.transpose(scaled_tens_tri,(3,0,1,2))
    scaled_tens_4_path = np.transpose(out_tens_tri,(3,0,1,2))

    geo_delta_t = 0.1#0.01#0.005
    #geo_iters = 6000 # 22000 for Kris annulus(delta_t=0.005), 32000 for cubic (delta_t=0.005)
    geo_iters = 1600 # 22000 for Kris annulus(delta_t=0.005), 32000 for cubic (delta_t=0.005)
    euler_delta_t = 0.1
    euler_iters = 4460 # 14600
    geox, geoy, geoz = geodesic.geodesicpath_3d(tens_4_path, out_mask,\
                                                start_coords[0], init_velocities[0], \
                                                geo_delta_t, iter_num=geo_iters)
    #threshgeox, threshgeoy, threshgeoz = geodesic.geodesicpath_3d(thresh_tens_4_path, intermed_results['differentiable_mask'],\
    #                                                           start_coords[0], init_velocities[0], \
    #                                                           geo_delta_t, iter_num=geo_iters)
    scalegeox, scalegeoy, scalegeoz = geodesic.geodesicpath_3d(scaled_tens_4_path, out_mask,\
                                                               start_coords[0], init_velocities[0], \
                                                               geo_delta_t, iter_num=geo_iters)
    eulx, euly, eulz = euler.eulerpath_3d(tens_4_path, out_mask,\
                                          start_coords[0], init_velocities[0], euler_delta_t, iter_num=euler_iters)

    for_mat_file[run_case]['tens_4_path'] = tens_4_path
    for_mat_file[run_case]['scaled_tens_4_path'] = scaled_tens_4_path
    for_mat_file[run_case]['orig_path'] = (geox, geoy, geoz)
    #for_mat_file[run_case]['thresh_path'] = (threshgeox, threshgeoy, threshgeoz)
    for_mat_file[run_case]['scaled_path'] = (scalegeox, scalegeoy, scalegeoz)
    for_mat_file[run_case]['euler_path'] = (eulx, euly, eulz)

  # end for each case
  savemat(outdir + 'cubic_results.mat',for_mat_file)

