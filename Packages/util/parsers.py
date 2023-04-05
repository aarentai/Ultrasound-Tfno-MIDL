# methods to parse output

# TODO is this the right place for these specific tools, or better in an analysis or vis area?

# This method extracts Iteration number, residual norm and minimum energy
# from output that has lines like:
# 28 KSP Residual norm 1.285554834550e+00
# Iter 28: current energy  is 59.4001
# Iter 28: minimum energy  is 26.4642
# 29 KSP Residual norm 1.276013902532e+00
# Iter 29: current energy  is 74.714
# Iter 29: minimum energy  is 26.4642
def parseMetricEstStdout(outstr):
  res_iter = []
  res_norm = []
  cur_iter = []
  cur_energy = []
  min_iter = []
  min_energy = []
  total_energy = -1
  lines = outstr.decode().split('\n')
  for line in lines:
    words = line.split()
    if len(words) == 5:
      # could be residual line or total energy line
      if ' '.join(words[1:4]) == 'KSP Residual norm':
        res_iter.append(int(words[0]))
        res_norm.append(float(words[4]))
      elif ''.join(words[1:3]) == 'total energy':
        # expect just one of these in outstr
        total_energy = float(words[4])
    elif len(words) == 6:
      # could be minimum energy line
      phrase = ' '.join(words[2:4])
      if phrase == 'minimum energy':
        min_iter.append(int(words[1].strip(':')))
        min_energy.append(float(words[5]))
      elif phrase == 'current energy':
        cur_iter.append(int(words[1].strip(':')))
        cur_energy.append(float(words[5]))
    else:
      # ignore
      continue
  result = {}
  result['KSP Residual norm'] = {'iter': res_iter, 'value': res_norm}
  result['minimum energy'] = {'iter': min_iter, 'value': min_energy}
  result['current energy'] = {'iter': cur_iter, 'value': cur_energy}
  result['total energy'] = total_energy
  return(result)
