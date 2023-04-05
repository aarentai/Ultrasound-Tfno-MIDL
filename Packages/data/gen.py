# Basic data generation methods, including filling/setting operations
from lazy_imports import sitk, np, torch
from . import constants as dc
import math

def zeroImage(img):
 img = img * 0.0

# TODO decide whether we like this way of importing sitk, numpy and/or torch
#      The idea is to only import them when we actually want to use them
def newImage(im_type, im_size, data_type, device=None):
  # Create a new image filled with zeros
  # works for n-d data
  dc.fill_data_type_map(im_type)
  if im_type == dc.IM_TYPE_SITK:
    #import SimpleITK as sitk
    im = sitk.Image(im_size, dc.data_type_map[data_type]["sitk"])
    zeroImage(im)
    return (im)
  elif im_type == dc.IM_TYPE_NP:
   #import numpy as np
   return np.zeros(im_size, dc.data_type_map[data_type]["np"])
  elif im_type == dc.IM_TYPE_TORCH:
    #import torch
    if device is None:
      dev = torch.device('cpu')
    else:
      dev = device
    return torch.tensor(newImage(dc.IM_TYPE_NP, im_size, data_type), device = dev)
  else:
    print("Cannot create image of type ", im_type, ", returning empty array")
    return[]


#############################
# Time-parameterized curves #
#############################

def cos_t(t): # circle_f
  ft = np.cos(t)
  return(ft)

def sin_t(t):
  gt = np.sin(t)
  return(gt)

def sin_2t(t):
  gt = np.sin(2*t)
  return(gt)

def r_cos_t(t, r=1): # circle_f
  ft = r * np.cos(t)
  return(ft)

def r_sin_t(t, r=1):
  gt = r * np.sin(t)
  return(gt)

def d_cos_t(t, dt=1):
  df = -np.sin(t)
  return(df / dt)

def d_sin_t(t, dt=1):
  dg = np.cos(t)
  return(dg / dt)
 
def d_sin_2t(t, dt=1):
  dh = 2*np.cos(2*t)
  return(dh / dt)

def d_r_cos_t(t, r=1, dt=1):
  df = -r * np.sin(t)
  return(df / dt)

def d_r_sin_t(t, r=1, dt=1):
  dg = r * np.cos(t)
  return(dg / dt)

def dd_cos_t(t, dt=1):
  ddf = -np.cos(t)
  return(ddf / (dt*dt))

def dd_sin_t(t, dt=1):
  ddg = -np.sin(t)
  return(ddg / (dt*dt))

def dd_r_cos_t(t, r=1, dt=1):
  ddf = -r * np.cos(t)
  return(ddf / (dt*dt))

def dd_r_sin_t(t, r=1, dt=1):
  ddg = -r * np.sin(t)
  return(ddg / (dt*dt))

 
def cubic(t, a3, a2, a1, a0):
  #ft = a3 * t**3 + a2 * t * t + a1 * t + a0
  ft = a3 * t*t*t + a2 * t * t + a1 * t + a0
  return(ft)

def d_cubic(t, a3, a2, a1, a0, dt):
  df = 3 * a3 * t * t + 2 * a2 * t + a1
  return(df / dt)

def dd_cubic(t, a3, a2, a1, a0):
  # wrong! divide by dt^2
  ddft = 6 * a3 * t + 2 * a2
  return(ddft)

def par_curv(k, ft, dft, gt, dgt):
  # Returns the positive and negative branches from http://mathworld.wolfram.com/ParallelCurves.html
  # k should be between 0 and 1
  denom = np.sqrt(dft*dft + dgt*dgt)
  xterm = (k * dgt) / denom
  yterm = (k * dft) / denom
  px = ft + xterm
  nx = ft - xterm
  py = gt - yterm
  ny = gt + yterm
  return(px,py,nx,ny)

def n_par_curv(k, ft, dft, ddft, gt, dgt, ddgt):
  # Returns the normals of the positive and negative branches from http://mathworld.wolfram.com/ParallelCurves.html
  # k should be between 0 and 1
  denom = dft*dft + dgt*dgt
  sqrt_denom = np.sqrt(denom)
  common_numer = dft * ddft + dgt * ddgt
  xterm = (sqrt_denom * k * ddft - (k * dft * common_numer) / sqrt_denom) / denom
  yterm = (sqrt_denom * k * ddgt - (k * dgt * common_numer) / sqrt_denom) / denom
  pdx = ft + xterm
  ndx = ft - xterm
  pdy = gt - yterm
  ndy = gt + yterm
  return(pdx,pdy,ndx,ndy)

def make_annulus(radius, width, nt):
 # make annulus of width (between 0 and 1)
 # and time resolution nt (number of time steps)
 # returns the outer (px, py) and inner (nx, ny) curves of the annulus.
 t = np.linspace(0, np.pi, nt)
 (px, py, nx, ny) = par_curv(width, r_cos_t(t, radius), d_r_cos_t(t, radius, nt),
                             r_sin_t(t, radius), d_r_sin_t(t, radius, nt))
 return ((px, py), (nx, ny))

def add_to_2D_image(im, xys, dxdys, ds, round_to_1=True):
  for xbin in ds.keys():
    for ybin in ds[xbin].keys():
      pixsum = 0
      cnt = 0
      for entry in ds[xbin][ybin]:
        pixsum += entry[2] # anti aliasing, by adding percent of pixel coverage instead of hard setting to 1 or 0
        cnt += 1
      if cnt > 0:
        im[xbin,ybin] = im[xbin,ybin] + pixsum / cnt
  if round_to_1:
    im[im > 0] = 1
  return(im)

def add_to_3D_image(im, xys, pixsum, cnt, round_to_1=True):
  for ii in range(im.shape[0]):
    for jj in range(im.shape[1]):
      for kk in range(im.shape[2]):
        if cnt[ii,jj,kk] > 0:
          im[ii,jj,kk] = im[ii,jj,kk] + pixsum[ii,jj,kk] / cnt[ii,jj,kk]
  if round_to_1:
    #print(np.max(im))
    im[im >= 0.05] = 1 # 0.05 is somewhat arbitrary but works better than im[im > 0] = 1
    im[im < 1] = 0
  else:
    # this a different version of rounding that works better for 3d annulus at the points away from center in the z direction
    im[pixsum > 1] = 1 
    im[pixsum <= 1] = 0
  return(im)

def add_to_2D_seed_image(im, xys, xrg, yrg, seed1, seed2):
  # Basically, set a seed at time 1 and at time 2, rest of image is 0
  # Here seed1 is the index into x,y at t1, and seed2 is the index into x,y at t2
  # There is probably a more elegant way to do this.
  # TODO decide if we want an antialiased version of the seed (ala add_to_2D_image)
  xinc =  (xrg[1]-xrg[0]) / im.shape[0]
  yinc =  (yrg[1]-yrg[0]) / im.shape[1]
  for c in range(xys.shape[0]):
    xi = xys[0, c, seed1]
    yj = xys[1, c, seed1]
    xidx = int(math.floor((xi-xrg[0]) / xinc))
    yidx = int(math.floor((yj-yrg[0]) / yinc))
    im[xidx,yidx] = 1
    xi = xys[0, c, seed2]
    yj = xys[1, c, seed2]
    xidx = int(math.floor((xi-xrg[0]) / xinc))
    yidx = int(math.floor((yj-yrg[0]) / yinc))
    im[xidx,yidx] = 1
  return(im)

def add_to_3D_seed_image(im, x, y, zs, dx, dy, rsq, xrg, yrg, zrg, seed1, seed2):
  # Basically, set a seed at time 1 and at time 2, rest of image is 0
  # Here seed1 is the index into x,y at t1, and seed2 is the index into x,y at t2
  # There is probably a more elegant way to do this.
  # TODO decide if we want an antialiased version of the seed (ala add_to_2D_image)
  xinc =  (xrg[1]-xrg[0]) / im.shape[0]
  yinc =  (yrg[1]-yrg[0]) / im.shape[1]
  zinc = xinc
  for zz in zs:
    cur_rsq = rsq - zz**2
    if cur_rsq <= 0:
      break
    cur_r  = np.sqrt(cur_rsq)
    # seed 1
    max_tsq = cur_rsq / (dx[seed1]**2 + dy[seed1]**2)
    max_t = np.sqrt(max_tsq)
    tspc = np.linspace(-max_t, max_t, 100)
    for tt in tspc:
      pts, pcts = get_3D_antialiased_points((x[seed1] + tt*dy[seed1] - xrg[0]) / xinc, (y[seed1] - tt*dx[seed1] - yrg[0]) / yinc, (zz - zrg[0]) / zinc)
      for pt, pct in zip(pts,pcts):
        xbin = pt[0]
        ybin = pt[1]
        zbin = pt[2]
            
        im[xbin,ybin,zbin] = 1

    # seed 2
    max_tsq = cur_rsq / (dx[seed2]**2 + dy[seed2]**2)
    max_t = np.sqrt(max_tsq)
    tspc = np.linspace(-max_t, max_t, 100)
    for tt in tspc:
      pts, pcts = get_3D_antialiased_points((x[seed2] + tt*dy[seed2] - xrg[0]) / xinc, (y[seed2] - tt*dx[seed2] - yrg[0]) / yinc, (zz - zrg[0]) / zinc)
      for pt, pct in zip(pts,pcts):
        xbin = pt[0]
        ybin = pt[1]
        zbin = pt[2]
            
        im[xbin,ybin,zbin] = 1
    
  return(im)


def add_2D_isotropic_background(im):
  # fill image with 2d isotropic tensors
  iso = np.array([1.0, 0, 1.0], dtype=np.float64)
  im[:] = iso
  return(im)

def add_3D_isotropic_background(im):
  # fill image with 3d isotropic tensors
  iso = np.array([1.0, 0, 0, 1.0, 0, 1.0], dtype=np.float64)
  im[:] = iso
  return(im)
 
def add_to_2D_tensor_image(im, xys, dxdys, ds, ratio, template_tensor=None):
  # ratio equals major axis / minor axis
  # from https://sciencing.com/vector-perpendicular-8419773.html
  if template_tensor is None:
    template = np.zeros((2,2),dtype=np.float64)
    template[0,0] = ratio
    template[1,1] = 1.0
  else:
    template = template_tensor
  R = np.zeros((2,2),dtype=np.float64)
  triu_idx = np.triu_indices(2) # to extract upper triangle

  dx = np.zeros((im.shape[0],im.shape[1]))
  dy = np.zeros((im.shape[0],im.shape[1]))

  for xbin in ds.keys():
    for ybin in ds[xbin].keys():
      xsum = 0
      ysum = 0
      cnt = 0
      for entry in ds[xbin][ybin]:
        xsum += entry[2] * dxdys[0,entry[0],entry[1]]
        ysum += entry[2] * dxdys[1,entry[0],entry[1]]
        cnt += 1
      if cnt > 0:
        dx[xbin,ybin] = xsum
        dy[xbin,ybin] = ysum

  for ii in range(im.shape[0]):
    for jj in range(im.shape[1]):
      dxij = dx[ii,jj]
      dyij = dy[ii,jj]
      if abs(dxij)>0 and abs(dyij)>0:
        denom = np.sqrt(dxij*dxij + dyij*dyij)
        if denom == 0:
          denom=1
        R[0,0] = dxij / denom
        R[1,0] = dyij / denom
        R[0,1] = -R[1,0]
        R[1,1] = R[0,0]
        im[ii,jj] = np.matmul(R, np.matmul(template, np.transpose(R)))[triu_idx]
  
  return(im)

def add_to_3D_tensor_image(im, xy, dx, dy, pixsum, cnts, ratio, template_tensor=None):
  # ratio equals major axis / minor axis
  # from https://sciencing.com/vector-perpendicular-8419773.html
  if template_tensor is None:
    template = np.zeros((3,3),dtype=np.float64)
    template[0,0] = ratio
    template[1,1] = 1.0
    template[2,2] = 1.0
  else:
    template = template_tensor
  R = np.zeros((3,3),dtype=np.float64)
  triu_idx = np.triu_indices(3) # to extract upper triangle

 
  for ii in range(im.shape[0]):
    for jj in range(im.shape[1]):
      for kk in range(im.shape[2]):
        if pixsum[ii,jj,kk] > 0:
          dxijk = dx[ii,jj,kk] / pixsum[ii,jj,kk]
          dyijk = dy[ii,jj,kk] / pixsum[ii,jj,kk]
          dzijk = 0
          denom = np.sqrt(dxijk*dxijk + dyijk*dyijk)# + dzijk*dzijk (dzijk=0, so don't need to add in)
          if denom == 0:
            denom=1
          R[0,0] = dxijk / denom
          R[1,0] = dyijk / denom
          R[0,1] = -R[1,0]
          R[1,1] = R[0,0]
          R[2,2] = 1  
          im[ii,jj,kk] = np.matmul(R, np.matmul(template, np.transpose(R)))[triu_idx]

  return(im)

 
def get_antialiased_points(x, y):
  # From https://en.wikipedia.org/wiki/Spatial_anti-aliasing#Simplest_approach_to_anti-aliasing
  pts = []
  pcts = []
  for roundedx in [math.floor(x), math.ceil(x)]:
    for roundedy in [math.floor(y), math.ceil(y)]:
      percent_x = 1 - abs(x - roundedx)
      percent_y = 1 - abs(y - roundedy)
      percent = percent_x * percent_y
      pts.append((int(roundedx), int(roundedy)))
      pcts.append(percent)
  return(pts, pcts)

def get_3D_antialiased_points(x, y, z):
  # From https://en.wikipedia.org/wiki/Spatial_anti-aliasing#Simplest_approach_to_anti-aliasing
  pts = []
  pcts = []
  xvals = [math.floor(x), math.ceil(x)]
  if xvals[1] == xvals[0]:
    xvals = [xvals[0]]
  yvals = [math.floor(y), math.ceil(y)]
  if yvals[1] == yvals[0]:
    yvals = [yvals[0]]
  zvals = [math.floor(z), math.ceil(z)]
  if zvals[1] == zvals[0]:
    zxvals = [zvals[0]]

  for roundedx in xvals:
    for roundedy in yvals:
      for roundedz in zvals:
        percent_x = 1 - abs(x - roundedx)
        percent_y = 1 - abs(y - roundedy)
        percent_z = 1 - abs(z - roundedz)
        percent = percent_x * percent_y * percent_z
        pts.append((int(roundedx), int(roundedy), int(roundedz)))
        pcts.append(percent)
  return(pts, pcts)
 
def gen_2D_tensor_image(xsz, ysz, tmin, tmax, numt, xf, dxf, yf, dyf, fdist, numf, ratio, seed_t1, seed_t2, do_isotropic=True, do_blurring=True, do_matching_range=False, xrng=None, yrng=None, template_tensor=None, zero_padding_width=None):
  # fdist should be between 0 and 1
  # ratio is the desired ratio between major and minor axes of each tensor
  # seed_t1 and seed_t2 are expressed as a fraction of the time range.  ie seed_t1 == 0.25 says to put a seed 1/4 of the way along the curve
  # if zero_padding_width is provided, make sure border of that width is all zeros in the final image
  t = np.linspace(tmin, tmax, numt, dtype=np.float64)
  x = xf(t)
  y = yf(t)
  print("(x0,y0)=",x[0],y[0])
  print("(x1,y1)=",x[-1],y[-1])
  dx = dxf(t, numt)
  dy = dyf(t, numt)

  # Convert seed_t1, seed_t2 into indices into t which is length numt
  seed1 = min(int(round(seed_t1 * (numt-1))), t.shape[0])
  seed2 = min(int(round(seed_t2 * (numt-1))), t.shape[0])

  (px, py, nx, ny) = par_curv(fdist, x, dx, y, dy)
  minx = min(min(px), min(nx))
  maxx = max(max(px), max(nx))
  miny = min(min(py), min(ny))
  maxy = max(max(py), max(ny))
  minxy = min(minx, miny)
  maxxy = max(maxx, maxy)
  if xrng is None:
    xrg = (minx-0.2, maxx+0.2)
  elif do_matching_range: 
    xrg = (minxy-0.2, maxxy+0.2)
  else:
    xrg = xrng
  if yrng is None:
    yrg = (miny-0.2, maxy+0.2)
  elif do_matching_range: 
    yrg = (minxy-0.2, maxxy+0.2)
  else:
    yrg = yrng

  print("xrg:", xrg)
  print("yrg:", yrg)
  if xrg != yrg:
    print('xrg and yrg are not equal.  In order to get proper geodesics, it is important they are equal.')
    minrng = min(xrg[0],yrg[0])
    maxrng = max(xrg[1],yrg[1])
    xrg = (minrng, maxrng)
    yrg = (minrng, maxrng)
    print('adjusting xrg, yrg to ', xrg, yrg)

  img = np.zeros((xsz, ysz))
  tens = np.zeros((xsz, ysz, 3),dtype=np.float64)
  seed = np.zeros((xsz, ysz))

  if do_isotropic:
    add_2D_isotropic_background(tens)

  # roll off the ratio by e^(a*d + b) where b = ln_ratio, a = -b / (fdist+finc)
  blurred_ratio = ratio
  finc = fdist / numf
  ln_ratio = math.log(ratio)
  #a = -ln_ratio / (fdist + finc) # exponential rolloff
  a = (1 - ratio) / (fdist + finc) # linear rolloff
  if do_blurring:
    #blurred_ratio = math.exp(a * fdist + ln_ratio) # exponential rolloff
    blurred_ratio = a * fdist + ratio # linear rolloff

  xs = []
  ys = []
  xs.append(px)
  xs.append(nx)
  ys.append(py)
  ys.append(ny)
  #add_to_2D_image(img, nx, ny, xrg, yrg)
  #add_to_2D_tensor_image(tens, nx, ny, np.gradient(nx), np.gradient(ny), xrg, yrg, blurred_ratio, template_tensor)
  #add_to_2D_seed_image(seed, nx, ny, xrg, yrg, seed1, seed2)
  #add_to_2D_image(img, px, py, xrg, yrg)
  #add_to_2D_tensor_image(tens, px, py, np.gradient(px), np.gradient(py), xrg, yrg, blurred_ratio, template_tensor)
  #add_to_2D_seed_image(seed, px, py, xrg, yrg, seed1, seed2)
  

  first_time = True
  for d in np.linspace(fdist, 0, numf, endpoint = False, dtype=np.float64):
    if first_time:
      first_time = False
      continue # we did this one already
  
    (px, py, nx, ny) = par_curv(d, x, dx, y, dy)
    if do_blurring:
      #blurred_ratio = math.exp(a * d + ln_ratio) # exponential rolloff
      blurred_ratio = a * d + ratio # linear rolloff
      #add_to_2D_image(img, nx, ny, xrg, yrg)
      #add_to_2D_tensor_image(tens, nx, ny, np.gradient(nx), np.gradient(ny), xrg, yrg, blurred_ratio, template_tensor)
      #add_to_2D_seed_image(seed, nx, ny, xrg, yrg, seed1, seed2)
      #add_to_2D_image(img, px, py, xrg, yrg)
      #add_to_2D_tensor_image(tens, px, py, np.gradient(px), np.gradient(py), xrg, yrg, blurred_ratio, template_tensor)
      #add_to_2D_seed_image(seed, px, py, xrg, yrg, seed1, seed2)
    xs.append(px)
    xs.append(nx)
    ys.append(py)
    ys.append(ny)

  xs.append(x)
  ys.append(y)

  xys = np.array((xs,ys))
  dxdys = np.zeros_like(xys)
  ds = {}
  xinc = (xrg[1]-xrg[0]) / xsz
  yinc = (yrg[1]-yrg[0]) / ysz
  for c in range(xys.shape[1]):
    dxdys[0,c,:] = np.gradient(xys[0,c,:])
    dxdys[1,c,:] = np.gradient(xys[1,c,:])

    for idx in range(xys.shape[-1]):
      pts, pcts = get_antialiased_points((xys[0,c,idx] - xrg[0]) / xinc, (xys[1,c,idx] - yrg[0]) / yinc)
      for pt, pct in zip(pts,pcts):
        xbin = pt[0]
        ybin = pt[1]
        if xbin not in ds.keys():
          ds[xbin] = {}
        if ybin not in ds[xbin].keys():
          ds[xbin][ybin] = []
        ds[xbin][ybin].append([c,idx,pct]) # c for which curve it is, pct for the amount of function value to include

  add_to_2D_image(img, xys, dxdys, ds)
  add_to_2D_tensor_image(tens, xys, dxdys, ds, ratio, template_tensor) # need to figure out blurring in this new context
  add_to_2D_seed_image(seed, xys, xrg, yrg, seed1, seed2)
        
  #add_to_2D_image(img, x, y, xrg, yrg)
  #add_to_2D_tensor_image(tens, x, y, dx, dy, xrg, yrg, ratio, template_tensor) # even with blurring, center of image should be full ratio
  #add_to_2D_seed_image(seed, x, y, xrg, yrg, seed1, seed2)

  if zero_padding_width:
    if do_isotropic:
      zero = np.ones((3))
      zero[1] = 0
    else:
      zero = np.zeros((3))
    tens[0:zero_padding_width,:,:] = zero
    tens[-zero_padding_width:,:,:] = zero
    tens[:,0:zero_padding_width,:] = zero
    tens[:,-zero_padding_width:,:] = zero
    img[0:zero_padding_width,:] = 0
    img[-zero_padding_width:,:] = 0
    img[:,0:zero_padding_width] = 0
    img[:,-zero_padding_width:] = 0
    seed[0:zero_padding_width,:] = 0
    seed[-zero_padding_width:,:] = 0
    seed[:,0:zero_padding_width] = 0
    seed[:,-zero_padding_width:] = 0

  return(img, tens, seed, xrg, yrg)

def gen_3D_tensor_image(xsz, ysz, tmin, tmax, numt, xf, dxf, yf, dyf, fdist, numf, ratio, seed_t1, seed_t2, do_isotropic=True, do_blurring=True, do_matching_range=False, xrng=None, yrng=None, zrng=None, template_tensor=None, zero_padding_width=None):
  # Compute 3D tensor image that is a tube version of the 2D tensor image from gen_2D_tensor_image
  # with the center slice in the z-direction the same as gen_2D_tensor_image (but with 3D tensors)
  # fdist should be between 0 and 1
  # ratio is the desired ratio between major and minor axes of each tensor
  # seed_t1 and seed_t2 are expressed as a fraction of the time range.  ie seed_t1 == 0.25 says to put a seed 1/4 of the way along the curve
  # if zero_padding_width is provided, make sure border of that width is all zeros in the final image
  t = np.linspace(tmin, tmax, numt, dtype=np.float64)
  x = xf(t)
  y = yf(t)
  
  #print("(x0,y0,z0)=",x[0],y[0],zcenter)
  #print("(x1,y1,z0)=",x[-1],y[-1],zcenter)
  dx = dxf(t, numt)
  dy = dyf(t, numt)

  # scale isotropic tensors outside image by this factor
  iso_scale = 0.1
  
  # Convert seed_t1, seed_t2 into indices into t which is length numt
  seed1 = min(int(round(seed_t1 * (numt-1))), t.shape[0])
  seed2 = min(int(round(seed_t2 * (numt-1))), t.shape[0])

  (px, py, nx, ny) = par_curv(fdist, x, dx, y, dy)
  minx = min(min(px), min(nx))
  maxx = max(max(px), max(nx))
  distx = max(abs(px-nx))
  miny = min(min(py), min(ny))
  maxy = max(max(py), max(ny))
  disty = max(abs(py-ny))
  minxy = min(minx, miny)
  maxxy = max(maxx, maxy)
  maxdist = max(distx, disty)
  if xrng is None:
    xrg = (minx-0.2, maxx+0.2)
  elif do_matching_range: 
    xrg = (minxy-0.2, maxxy+0.2)
  else:
    xrg = xrng
  if yrng is None:
    yrg = (miny-0.2, maxy+0.2)
  elif do_matching_range: 
    yrg = (minxy-0.2, maxxy+0.2)
  else:
    yrg = yrng
  if zrng is None:
    zrg = (-maxdist-0.2, maxdist+0.2)
  elif do_matching_range: 
    zrg = (minxy-0.2, maxxy+0.2)
  else:
    zrg = zrng
    

  print("xrg:", xrg)
  print("yrg:", yrg)
  print("zrg:", zrg)
  if xrg != yrg:
    print('xrg and yrg are not equal.  In order to get proper geodesics, it is important they are equal.')
    minrng = min(xrg[0],yrg[0])
    maxrng = max(xrg[1],yrg[1])
    xrg = (minrng, maxrng)
    yrg = (minrng, maxrng)
    print('adjusting xrg, yrg to ', xrg, yrg)

  xinc = (xrg[1]-xrg[0]) / xsz
  yinc = (yrg[1]-yrg[0]) / ysz
  zinc = xinc #(zrg[1]-zrg[0]) / zsz
  zsz = int(round((zrg[1]-zrg[0]) / zinc))
  zcenter = int(zsz / 2)

  img = np.zeros((xsz, ysz, zsz))
  tens = np.zeros((xsz, ysz, zsz, 6),dtype=np.float64)
  seed = np.zeros((xsz, ysz, zsz))

  if do_isotropic:
    add_3D_isotropic_background(tens)
    tens = iso_scale * tens

  # roll off the ratio by e^(a*d + b) where b = ln_ratio, a = -b / (fdist+finc)
  blurred_ratio = ratio
  finc = fdist / numf
  ln_ratio = math.log(ratio)
  #a = -ln_ratio / (fdist + finc) # exponential rolloff
  a = (1 - ratio) / (fdist + finc) # linear rolloff
  if do_blurring:
    #blurred_ratio = math.exp(a * fdist + ln_ratio) # exponential rolloff
    blurred_ratio = a * fdist + ratio # linear rolloff


  xys = np.array((x,y))
  dxdys = np.array((dx,dy))

  rsq=(px[0]-x[0])**2 + (py[0]-y[0])**2
  rsq2=(nx[0]-x[0])**2 + (ny[0]-y[0])**2
  radius = np.sqrt(rsq)
  radius2 = np.sqrt(rsq2)
 
  zctrval = zrg[0] + zinc * zcenter
  #zs = np.linspace(-radius, radius, zsz)
  # Do this way to ensure we're symmetric in the z-direction
  zhalf = np.linspace(-radius,0,int((zsz-1)/2),endpoint=False)
  zs = np.concatenate((zhalf,[0]))  
  idx=0

  #xcurval = x[idx]
  #ycurval = y[idx]

  dxim = np.zeros((tens.shape[0],tens.shape[1],tens.shape[2]))
  dyim = np.zeros((tens.shape[0],tens.shape[1],tens.shape[2]))
  pixsum = np.zeros((tens.shape[0],tens.shape[1],tens.shape[2]))
  cnts = np.zeros((tens.shape[0],tens.shape[1],tens.shape[2]))

  #print(xcurval, ycurval)
  for zz in zs:
    cur_rsq = rsq - zz**2
    if cur_rsq <= 0:
      continue 
    cur_r  = np.sqrt(cur_rsq)
    for idx in range(len(x)):
      max_tsq = cur_rsq / (dx[idx]**2 + dy[idx]**2)
      max_t = np.sqrt(max_tsq)
      #tspc = np.linspace(-max_t, max_t, 100)
      thalf = np.linspace(-max_t,0,75,endpoint=False)
      tspc = np.concatenate((thalf,[0],-thalf[::-1]))
      for tt in tspc:
        pts, pcts = get_3D_antialiased_points((x[idx] + tt*dy[idx] - xrg[0]) / xinc, (y[idx] - tt*dx[idx] - yrg[0]) / yinc, (zz - zrg[0]) / zinc)
        for pt, pct in zip(pts,pcts):
          xbin = pt[0]
          ybin = pt[1]
          zbin = pt[2]
            
          ozbin = zcenter + (zcenter - zbin)
            
          dxim[xbin,ybin,zbin] += pct * dxdys[0,idx]
          dyim[xbin,ybin,zbin] += pct * dxdys[1,idx]
          pixsum[xbin,ybin,zbin] += pct
          cnts[xbin,ybin,zbin] += 1
          if ozbin != zbin:
            dxim[xbin,ybin,ozbin] += pct * dxdys[0,idx]
            dyim[xbin,ybin,ozbin] += pct * dxdys[1,idx]
            pixsum[xbin,ybin,ozbin] += pct
            cnts[xbin,ybin,ozbin] += 1

  add_to_3D_image(img, xys, pixsum, cnts, round_to_1=False)
  add_to_3D_tensor_image(tens, xys, dxim, dyim, pixsum, cnts, ratio, template_tensor) # need to figure out blurring in this new context
  add_to_3D_seed_image(seed, x, y, zs, dx, dy, rsq, xrg, yrg, zrg, seed1, seed2)

  if zero_padding_width:
    if do_isotropic:
      zero = iso_scale * np.ones((6))
      zero[1] = 0
      zero[2] = 0
      zero[4] = 0
    else:
      zero = np.zeros((6))
    tens[0:zero_padding_width,:,:,:] = zero
    tens[-zero_padding_width:,:,:,:] = zero
    tens[:,0:zero_padding_width,:,:] = zero
    tens[:,-zero_padding_width:,:,:] = zero
    tens[:,:,0:zero_padding_width,:] = zero
    tens[:,:,-zero_padding_width:,:] = zero
    img[0:zero_padding_width,:,:] = 0
    img[-zero_padding_width:,:,:] = 0
    img[:,0:zero_padding_width,:] = 0
    img[:,-zero_padding_width:,:] = 0
    img[:,:,0:zero_padding_width] = 0
    img[:,:,-zero_padding_width:] = 0
    seed[0:zero_padding_width,:,:] = 0
    seed[-zero_padding_width:,:,:] = 0
    seed[:,0:zero_padding_width,:] = 0
    seed[:,-zero_padding_width:,:] = 0
    seed[:,:,0:zero_padding_width] = 0
    seed[:,:,-zero_padding_width:] = 0

  return(img, tens, seed, xrg, yrg, zrg, zsz)

 
def tensor_to_uv(img):
  xsize=img.shape[0]
  ysize=img.shape[1]
  u = np.zeros((xsize, ysize), dtype=np.float64)
  v = np.zeros((xsize, ysize), dtype=np.float64)
  tens = np.zeros((2,2), dtype=np.float64)
  triu_idx = np.triu_indices(2)
  pos_vec = np.zeros((2,1), dtype=np.float64)
  for x in range(xsize):
    for y in range(ysize):
     tens[triu_idx] = img[x,y]
     tens[1,0] = tens[0,1]
     evals, evecs = np.linalg.eigh(tens)
     if abs(evecs[1][0]) > abs(evecs[1][1]):
       if evecs[1][0] >= 0:
         pos_vec = evecs[1]
       else:
         pos_vec = -evecs[1]
     else:
       if evecs[0][1] >= 0:
         pos_vec = evecs[1]
       else:
         pos_vec = -evecs[1]
     #if evecs[1][0] >= 0:
     #  pos_vec = evecs[1]
     #else:
     #  pos_vec = -evecs[1]
     u[x,y] = evals[1] * pos_vec[0]
     v[x,y] = evals[1] * pos_vec[1]
  return(u,v)

def gen_2D_annulus(xsz, ysz, ratio, do_isotropic=True, do_blurring=True,
                   do_matching_range=False, xrng=None, yrng=None, template_tensor=None,zero_padding_width=None):
  return(gen_2D_tensor_image(xsz, ysz, 0, np.pi, 1000, cos_t, d_cos_t, sin_t, d_sin_t, 1/5.0, 15, ratio, 0.05, 0.95,
                             do_isotropic, do_blurring, do_matching_range, xrng=xrng, yrng=yrng, template_tensor=template_tensor,
                             zero_padding_width=zero_padding_width))

def gen_3D_annulus(xsz, ysz, ratio, do_isotropic=True, do_blurring=True,
                   do_matching_range=False, xrng=None, yrng=None, zrng=None, template_tensor=None,zero_padding_width=None):
  return(gen_3D_tensor_image(xsz, ysz, 0, np.pi, 1000, cos_t, d_cos_t, sin_t, d_sin_t, 1/5.0, 15, ratio, 0.05, 0.95,
                             do_isotropic, do_blurring, do_matching_range, xrng=xrng, yrng=yrng, zrng=zrng, template_tensor=template_tensor, zero_padding_width=zero_padding_width))

def gen_2D_rectangle_gradient_ratio(xsz, ysz, ratio, rotation=None, do_isotropic=True, zero_padding_width=None):
  # generate a 2D rectangle whose tensors vary from isotropic to ratio in the x direction
  # and vary in rotation between 0 and 90 degrees in the y direction
  if zero_padding_width is None:
    zero_padding_width = 0
  rect_tens = np.ones((xsz, ysz, 3),dtype=np.float64)
  rect_tens[:,:,1] = 0
  ev1 = 1
  vals = np.linspace(ev1, ratio, xsz - 2*zero_padding_width)
  tens = np.zeros((2,2))
  angles = np.linspace(0, np.pi / 2, ysz - 2*zero_padding_width)
  R = np.zeros((2,2))
  triu_idx = np.triu_indices(2) # to extract upper triangle
  
  const_R = np.zeros((2,2))
  const_R[0,0] = 1
  const_R[1,1] = 1
  if rotation:
    const_R[0,0] = np.cos(rotation)
    const_R[1,0] = np.sin(rotation)
    const_R[0,1] = -const_R[1,0]
    const_R[1,1] = const_R[0,0]
    
  for ii in range(zero_padding_width, xsz - zero_padding_width):
    tens[0,0] = ev1
    tens[1,0] = 0
    tens[0,1] = 0
    tens[1,1] = vals[ii-zero_padding_width]
    tens = np.matmul(const_R, np.matmul(tens, np.transpose(const_R)))
    rect_tens[ii,zero_padding_width] = tens[triu_idx]
    for jj in range(zero_padding_width+1, ysz - zero_padding_width):
      R[0,0] = np.cos(angles[jj-zero_padding_width])
      R[1,0] = np.sin(angles[jj-zero_padding_width])
      R[0,1] = -R[1,0]
      R[1,1] = R[0,0]
      rect_tens[ii,jj] = np.matmul(R, np.matmul(tens, np.transpose(R)))[triu_idx]
  return(rect_tens)

def gen_2D_rectangle_constant_ratio(xsz, ysz, ratio, rotation=None, do_isotropic=True, zero_padding_width=None):
  # generate a 2D rectangle whose tensors are 1 to ratio everywhere
  if zero_padding_width is None:
    zero_padding_width = 0
    
  rect_tens = np.ones((xsz, ysz, 3),dtype=np.float64)
  rect_tens[:,:,1] = 0
  if rotation:
    tens = np.zeros((2,2))
    const_R = np.zeros((2,2))
    triu_idx = np.triu_indices(2) # to extract upper triangle

    if ratio < 1:
      rect_tens = ratio * rect_tens # make isotropic same scale as smaller eigenvalue

    tens[0,0] = 1
    tens[1,0] = 0
    tens[0,1] = 0
    tens[1,1] = ratio

    const_R[0,0] = np.cos(rotation)
    const_R[1,0] = np.sin(rotation)
    const_R[0,1] = -const_R[1,0]
    const_R[1,1] = const_R[0,0]

    rect_tens[zero_padding_width:-zero_padding_width,zero_padding_width:-zero_padding_width] = np.matmul(const_R, np.matmul(tens, np.transpose(const_R)))[triu_idx]
   
  else:
    if ratio < 1:
      rect_tens = ratio * rect_tens # make isotropic same scale as smaller eigenvalue
      rect_tens[zero_padding_width:-zero_padding_width,zero_padding_width:-zero_padding_width,0] = 1
    else: 
      rect_tens[zero_padding_width:-zero_padding_width,zero_padding_width:-zero_padding_width,2] = ratio

  return(rect_tens)     


if __name__ == "__main__":
 circ_rng = (-1.4,1.4)
 img, tens, seed, xrg, yrg = gen_2D_annulus(100, 100, 6.0, True, False, False, xrng = circ_rng, yrng = circ_rng)
 
