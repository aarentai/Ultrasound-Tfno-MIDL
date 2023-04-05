import pickle
from sims import metricEstSim as mes

# to avoid unittest, run from top level metpy directory as
# PYTHONPATH=. python3 test/testMetricEstSim.py

if __name__ == "__main__":
  #indir = '/usr/sci/projects/HCP/Kris/NSFCRCNS/TestData/3D/'
  inroot = '/usr/sci/projects/HCP/Kris/NSFCRCNS/prepped_data/'
  outdir = '/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/working_3d_python/simulation_results/'
  #for_mat_file = {}
  cases = []
  all_start_coords = []
  cases.append('103818')
  all_start_coords.append([[62,126,56]])
  cases.append('105923')
  all_start_coords.append([[61,125,56]])
  cases.append('111312')
  all_start_coords.append([[62,128,56]])
  cases.append('103818up2')
  all_start_coords.append([[124,252,112]])
  cases.append('105923up2')
  all_start_coords.append([[122,250,112]])
  cases.append('111312up2')
  all_start_coords.append([[124,256,112]])

  init_velocities = [None]

  #min_evals = [5e-2,1e-2,5e-3,1e-3,5e-10]
  #min_evals = [5e-3]
  #num_iters = [50, 450, 1000, 3000]
  #sigmas = [None, 1.5]
  min_evals = [5e-3]
  num_iters = [3000]
  sigmas = [1.5]

  for (cc, start_coords) in zip(cases, all_start_coords):
    run_case = f'{cc}'
    print(f"Running simulation for {run_case}")
    subj = run_case[:6]
    indir = f"{inroot}{subj}/"

    if "up2" in run_case:
      tens_file = 'dti_1000_up2_tensor.nhdr'
      mask_file = 'dti_1000_up2_FA_mask.nhdr'
    else:
      tens_file = 'dti_1000_tensor.nhdr'
      mask_file = 'dti_1000_FA_mask.nhdr'

    output_root_dir = f"{outdir}{run_case}"
    results = mes.run_simulation(run_case, tens_file, mask_file, indir, output_root_dir, min_evals, num_iters, sigmas, start_coords, init_velocities)
    with open(f'{output_root_dir}/results.pkl', 'wb') as f:
      pickle.dump(results, f)  

  #mes.summarize_results(results, run_case, output_root_dir, 1, 1, use_minimum_energy)

  
