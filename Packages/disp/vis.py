# for data vis methods
# When using itkwidgets, must import itk directly.  It doesn't play nice with lazy_imports
import itk
from lazy_imports import sitk, np, linalg
from lazy_imports import plt, PatchCollection, Ellipse, EllipseCollection
from lazy_imports import ipywidgets as widg
from lazy_imports import itkview #itkwidgets.view
from lazy_imports import pv # pyvista
from lazy_imports import interactive
from lazy_imports import ipywidgets
from data.convert import GetSITKImageFromNP, GetNPArrayFromSITK, GetNPArrayViewFromSITK
from data.io import readRaw

import algo.geodesic as geo
import algo.euler as euler

# TODO worth breaking into separate files based on image type?
#      ie vis_sitk.py, vis_np.py etc?

##########################################
# Simple ITK image visualization methods #
##########################################
def show_metadata(img):
 for key in img.GetMetaDataKeys():
   print ("\"{0}\":\"{1}\"".format(key, image.GetMetaData(key)))

def show_info(img):
 print("size: {0}".format(img.GetSize()))
 print("origin: {0}".format(img.GetOrigin()))
 print("spacing: {0}".format(img.GetSpacing()))
 print("direction: {0}".format(img.GetDirection()))
 print("dimension: {0}".format(img.GetDimension()))
 print("pixel type: {0}".format(img.GetPixelIDTypeAsString()))

#def show_slice(img, slc_no, ax):
#  if ax == 0:
#    slc = GetNPArrayViewFromSITK(img)[slc_no, :, :]
#  elif ax == 1:
#    slc = GetNPArrayViewFromSITK(img)[:, slc_no, :]
#  else:
#    slc = GetNPArrayViewFromSITK(img)[:, :, slc_no]

#    ax = plt.imshow(slc)
#    ax.set_cmap("gray")

def show_slice(img, slc_no, ax, title=None, margin=0.05, dpi=80, has_component_data=False):
  if ax == 0:
    show_2d(img[slc_no, :, :], title+f", Axis {ax}, Slice {slc_no}", margin, dpi, has_component_data)
  elif ax == 1:
    show_2d(img[:, slc_no, :], title+f", Axis {ax}, Slice {slc_no}", margin, dpi, has_component_data)
  else:
    show_2d(img[:, :, slc_no], title+f", Axis {ax}, Slice {slc_no}", margin, dpi, has_component_data)

def show_2d(img, title=None, margin=0.05, dpi=80, has_component_data=False):
  if type(img) == np.ndarray:
    nda = img
    spacing = [1,1]
  else: 
    nda = GetNPArrayFromSITK(img, has_component_data)
    spacing = img.GetSpacing()
    
  if nda.ndim == 3:
    # fastest dim, either component or x
    c = nda.shape[-1]    
    # the the number of components is 3 or 4 consider it an RGB image
    if not c in (3,4):
      nda = nda[nda.shape[0]//2,:,:]
  
  elif nda.ndim == 4:
    c = nda.shape[-1]
    
    if not c in (3,4):
      raise Runtime("Unable to show 3D-vector Image")
      
    # take a z-slice
    nda = nda[nda.shape[0]//2,:,:,:]

  #ndat = nda.T      
  ysize = nda.shape[0]
  xsize = nda.shape[1]
    
  # Make a figure big enough to accommodate an axis of xpixels by ypixels
  # as well as the ticklabels, etc...
  if xsize > dpi and ysize > dpi:
    figsize = (1 + margin) * ysize / dpi, (1 + margin) * xsize / dpi
  else:
    figsize = (1 + margin) * dpi / ysize, (1 + margin) * dpi / xsize
  #print("fig size: ", figsize)

  #fig = plt.figure(figsize=figsize, dpi=dpi)
  fig = plt.figure()
  # Make the axis the right size...
  ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])
  
  #extent = (0, xsize*spacing[1], ysize*spacing[0], 0)
  # use following extent when transposing nda array.  previous extent when not transposing
  extent = (0, xsize*spacing[0], ysize*spacing[1], 0)
  
  t = ax.imshow(nda,extent=extent,interpolation=None)
  # use following imshow when transposing nda array.  previous imshow when not transposing
  #t = ax.imshow(ndat,extent=extent,interpolation=None, origin="upper")
  
  if nda.ndim == 2:
    t.set_cmap("gray")
  
  if(title):
    plt.title(title)

def show_2d_tensors(img, scale=1, title=None, margin=0.05, dpi=80, has_component_data=False):
  if type(img) == np.ndarray:
    nda = img
    spacing = [1,1]
  else: 
    nda = GetNPArrayFromSITK(img, has_component_data)
    spacing = img.GetSpacing()
      
  if nda.ndim == 3:
    # fastest dim, either component or x
    c = nda.shape[-1]        
    # the number of component is 3 consider it a tensor image
    if c != 3:
      raise Runtime("Unable to show 3D-vector Image")
          
    # take a z-slice
    #nda = nda[nda.shape[0]//2,:,:,:]
    #nda = nda[nda.shape[0]//2,:,:]
          
  xsize = nda.shape[0]
  ysize = nda.shape[1]
    
  # Make a figure big enough to accommodate an axis of xpixels by ypixels
  # as well as the ticklabels, etc...
  if xsize > dpi and ysize > dpi:
    figsize = (1 + margin) * ysize / dpi, (1 + margin) * xsize / dpi
  else:
    figsize = (1 + margin) * dpi / ysize, (1 + margin) * dpi / xsize
  #print("fig size: ", figsize)

  #fig = plt.figure(figsize=figsize, dpi=dpi)
  fig = plt.figure()
  # Make the axis the right size...
  ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])
  
  extent = (0, xsize*spacing[1], ysize*spacing[0], 0)
  ellipses = []

  if nda.shape[2] == 3:
    tens = np.zeros((nda.shape[0],nda.shape[1],2,2))    
    tens[:,:,0,0] = nda[:,:,0]
    tens[:,:,0,1] = nda[:,:,1]
    tens[:,:,1,0] = nda[:,:,1]
    tens[:,:,1,1] = nda[:,:,2]
    evals, evecs = np.linalg.eigh(tens)
  else:
    evals, evecs = np.linalg.eigh(nda)
  angles = np.degrees(np.arctan2(evecs[:,:,1,1],evecs[:,:,1,0]))
 
  scaled_evals = scale * evals

  ellipses = [Ellipse((x,y), width=scaled_evals[x,y,1], height = scaled_evals[x,y,0], angle=angles[x,y]) for y in range(ysize) for x in range(xsize)]

  # TODO try EllipseCollection
  collection = PatchCollection(ellipses, alpha=0.7)
  ax.add_collection(collection)
  ax.set_xlim(-1,xsize)
  ax.set_ylim(-1,ysize)
  ax.set_aspect('equal')
  
  #t = ax.imshow(nda,extent=extent,interpolation=None)
  
  #if nda.ndim == 2:
    #t.set_cmap("gray")
  
  if(title):
    plt.title(title)
  # For some reason, returning the fig causes it to appear twice in jupyter notebook
  #return(fig)


def show_3d_tensors(img, scale=1, title=None, margin=0.05, dpi=80, has_component_data=False):
  print('Not implemented yet.  See view_3d_tensors instead')
# Some code from https://stackoverflow.com/questions/41955492/how-to-plot-efficiently-a-large-number-of-3d-ellipsoids-with-matplotlib-axes3d
# To plot 3D ellipsoids in matplotlib
# number of ellipsoids 
# ellipNumber = 10

# #set colour map so each ellipsoid as a unique colour
# norm = colors.Normalize(vmin=0, vmax=ellipNumber)
# cmap = cm.jet
# m = cm.ScalarMappable(norm=norm, cmap=cmap)

# #compute each and plot each ellipsoid iteratively
# for indx in xrange(ellipNumber):
#     # your ellispsoid and center in matrix form
#     A = np.array([[np.random.random_sample(),0,0],
#                   [0,np.random.random_sample(),0],
#                   [0,0,np.random.random_sample()]])
#     center = [indx*np.random.random_sample(),indx*np.random.random_sample(),indx*np.random.random_sample()]

#     # find the rotation matrix and radii of the axes
#     U, s, rotation = linalg.svd(A)
#     radii = 1.0/np.sqrt(s) * 0.3 #reduce radii by factor 0.3 

#     # calculate cartesian coordinates for the ellipsoid surface
#     u = np.linspace(0.0, 2.0 * np.pi, 60)
#     v = np.linspace(0.0, np.pi, 60)
#     x = radii[0] * np.outer(np.cos(u), np.sin(v))
#     y = radii[1] * np.outer(np.sin(u), np.sin(v))
#     z = radii[2] * np.outer(np.ones_like(u), np.cos(v))

#     for i in range(len(x)):
#         for j in range(len(x)):
#             [x[i,j],y[i,j],z[i,j]] = np.dot([x[i,j],y[i,j],z[i,j]], rotation) + center


#     ax.plot_surface(x, y, z,  rstride=3, cstride=3,  color=m.to_rgba(indx), linewidth=0.1, alpha=1, shade=True)
# plt.show()

def view_3d_paths(img, paths=None, labels=None,xrng=None, yrng=None, zrng=None, viewer=None, num_tube_pts=500, tube_radius=0.5, colors=[]):
  if xrng is None:
    xrng = [0, img.shape[0]]
  if yrng is None:
    yrng = [0, img.shape[1]]
  if zrng is None:
    zrng = [0, img.shape[2]]
  if labels is None:
    labels = range(len(paths))

  # wrapping in do_view in case we want to add an interaction to update xrng,yrng,zrng  
  def do_view(img,paths,labels,xrng,yrng,zrng,colors,viewer=None):  
    #glyphs = []
    glyphs = {}
    if paths is not None:
      for p, label in zip(paths, labels):
        tube = path_to_tube(p[0], p[1], p[2], num_tube_pts, tube_radius)
        #glyphs.append(tube)
        glyphs[label] = tube

    if viewer:
      if img is not None:
        viewer.image = img[xrng[0]:xrng[1],yrng[0]:yrng[1],zrng[0]:zrng[1]]
        viewer.geometries = glyphs
    else:
      if img is not None:
        viewer = itkview(img[xrng[0]:xrng[1],yrng[0]:yrng[1],zrng[0]:zrng[1]],
                         geometries=glyphs, geometry_colors=colors)
      else:
        viewer = itkview(geometries=glyphs, geometry_colors=colors)
    return(viewer)

  viewer = do_view(img, paths, labels, xrng, yrng, zrng, colors, viewer)
 
  return(viewer)

def view_3d_tensors(tens, mask=None, img=None, paths=None, xrng=None, yrng=None, zrng=None, stride=5, scale=None, viewer=None, has_component_data=False, num_tube_pts=500, tube_radius=0.5):
  #if type(tens) == np.ndarray:
  #  nda = tens
  #  spacing = [1,1,1]
  #else: 
  #  nda = GetNPArrayFromSITK(tens, has_component_data)
  #  spacing = tens.GetSpacing()
 
  if xrng is None:
    xrng = [0, mask.shape[0]]
  if yrng is None:
    yrng = [0, mask.shape[1]]
  if zrng is None:
    zrng = [0, mask.shape[2]]

  sml_img = None
  if img is not None:
    sml_img = np.zeros_like(img)
    sml_img[xrng[0]:xrng[1],yrng[0]:yrng[1],zrng[0]:zrng[1]] = img[xrng[0]:xrng[1],yrng[0]:yrng[1],zrng[0]:zrng[1]]
    
  # wrapping in do_view in case we want to add an interaction to update xrng,yrng,zrng  
  def do_view(tens,mask,img,paths,xrng,yrng,zrng,stride,scale,viewer=None):  
    glyphs = tensors_to_mesh(tens, mask, xrng, yrng, zrng, stride, scale)
    if paths is not None:
      for p in paths:
        tube = path_to_tube(p[0], p[1], p[2], num_tube_pts, tube_radius)
        glyphs.append(tube)

    if viewer:
      if img is not None:
        viewer.image = sml_img
        viewer.geometries = glyphs
    else:
      if img is not None:
        viewer = itkview(sml_img,
                         geometries=glyphs)
      else:
        viewer = itkview(geometries=glyphs)
    return(viewer)

  viewer = do_view(tens, mask, img, paths, xrng, yrng, zrng, stride, scale, viewer)
 
  return(viewer)
 
def path_to_tube(pathx, pathy, pathz, num_tube_pts = 500, radius = 0.5):
  # swap x and z coordinates to match convention of itkwidgets.view
  spline=pv.Spline(np.column_stack((pathz, pathy, pathx)), num_tube_pts)
  #spline["scalars"] = np.arange(spline.n_points)
  tube = spline.tube(radius=radius)
  return(tube)
  
def tensors_to_mesh(tensor_field, mask, xrng=None, yrng=None, zrng=None, stride=1, scale=None):
  # convert tensors to pyvista PolyData Ellipsoids.  If scale is None, normalize the ellipses,
  # otherwise scale unnormalized ellipses by scale.
  # Ellipses provided for each voxel, striding in xrng, yrng, zrng where mask == 1
  if xrng is None:
    xrng = [0,mask.shape[0]]
  if yrng is None:
    yrng = [0, mask.shape[1]]
  if zrng is None:
    zrng = [0, mask.shape[2]]

  ptlist = []
  for x in range(xrng[0],xrng[1],stride):
    for y in range(yrng[0],yrng[1],stride):
      for z in range(zrng[0],zrng[1],stride):
        if (mask is None) or mask[x,y,z]:
          ptlist.append(np.array([x,y,z]))
  pts = np.array(ptlist)
  numpts = pts.shape[0]
  print('numpts',numpts)
  tfm_pts = np.copy(pts)
  # swap x and z axes to match convention of itkwidgets.view
  tfm_pts[:,0] = tfm_pts[:,2]
  tfm_pts[:,2] = pts[:,0]  
  point_cloud = pv.PolyData(tfm_pts)
  point_cloud.point_arrays['scalars'] = np.arange(numpts)

  tens = np.zeros((numpts,3,3))
  tens[:,0,0] = tensor_field[pts[:,0],pts[:,1],pts[:,2],0]
  tens[:,0,1] = tensor_field[pts[:,0],pts[:,1],pts[:,2],1]
  tens[:,0,2] = tensor_field[pts[:,0],pts[:,1],pts[:,2],2]
  tens[:,1,0] = tens[:,0,1]
  tens[:,1,1] = tensor_field[pts[:,0],pts[:,1],pts[:,2],3]
  tens[:,1,2] = tensor_field[pts[:,0],pts[:,1],pts[:,2],4]
  tens[:,2,0] = tens[:,0,2]
  tens[:,2,1] = tens[:,1,2]
  tens[:,2,2] = tensor_field[pts[:,0],pts[:,1],pts[:,2],5]
  evals, evecs = np.linalg.eigh(tens)

  if scale is not None:
    print ('smallest,largest max eigenvalue',np.min(evals[:,2]), np.max(evals[:,2]))
  #ellipses = [pv.ParametricEllipsoid(evals[p,2]/evals[p,2], evals[p,1]/evals[p,2], evals[p,0]/evals[p,2], center=pts[p], direction=evals[p,2]) for p in range(numpts)]
  ellipses = []
  
  # list comprehension is probably faster, can switch if don't get exceptions when converting real brain data
  for p in range(numpts):
    try:
      if scale is None:
        # This normalizes all ellipses, dangerous if need to compare relative sizes
        ellipses.append(pv.ParametricEllipsoid(evals[p,2]/evals[p,2], evals[p,1]/evals[p,2], evals[p,0]/evals[p,2], 
                                               center=tfm_pts[p], direction=[evecs[p,2,2],evecs[p,1,2],evecs[p,0,2]]))        
      else:
        # This keeps relative size differences across ellipses, and just scales to fit in window
        ellipses.append(pv.ParametricEllipsoid(evals[p,2]/scale, evals[p,1]/scale, evals[p,0]/scale, 
                                               center=tfm_pts[p], direction=[evecs[p,2,2],evecs[p,1,2],evecs[p,0,2]]))   
    except:
      print("error for point",p)
      print("tens",tens[p])
      print('evals',evals[p,2],evals[p,1],evals[p,0])
      print('evecs',evecs[p,2])
      print('pt',pts[p])
      break
  # Following puts glyphs at wrong points, so return ellipses directly
  #glyphs = point_cloud.glyph(geom=ellipses, indices=np.arange(numpts),scale=False, factor=1, rng=(0, numpts-1))
  #return(glyphs)
  return(ellipses)

  
def show_tensor_slice(img, slc_no, ax, scale=1, title=None, margin=0.05, dpi=80, has_component_data=False):
  if ax == 0:
    show_3d_tensors(img[slc_no, :, :], scale, title+f", Axis {ax}, Slice {slc_no}", margin, dpi, has_component_data)
  elif ax == 1:
    show_3d_tensors(img[:, slc_no, :], scale, title+f", Axis {ax}, Slice {slc_no}", margin, dpi, has_component_data)
  else:
    show_3d_tensors(img[:, :, slc_no], scale, title+f", Axis {ax}, Slice {slc_no}", margin, dpi, has_component_data)

def show_tensor_xslice(img, slc_no, scale=1, title=None, margin=0.05, dpi=80, has_component_data=False):
  if type(img) == np.ndarray:
    num_x = img.shape[0]
  else: 
    num_x = GetNPArrayViewFromSITK(img, has_component_data).shape[0]

  def next_x(slc):
    show_tensor_slice(img, slc, 0, scale, title, margin, dpi, has_component_data)
    fig = plt.gcf()
    fig.cur_slice = slc

  next_x(slc_no)
  
  def press(event):
    #print('press', event.key)
    sys.stdout.flush()
    if event.key == 'left':
      slc = max(fig.cur_slice-1,0)
      next_x(slc)
    elif event.key == 'right':
      slc = min(fig.cur_slice+1,num_x)
      next_x(slc)

  plt.gcf().canvas.mpl_connect('key_press_event', press)

  w = widg.IntSlider(value=slc_no,
                     min=0,
                     max=num_x,
                     step=1,
                     description='Slice Position:')

  widg.interactive(next_x, slc=w)



def show_tensor_yslice(img, slc_no, scale=1, title=None, margin=0.05, dpi=80, has_component_data=False):
  def next_y(slc):
    show_tensor_slice(img, slc, 1, scale, title, margin, dpi, has_component_data)

    
  if type(img) == np.ndarray:
    num_y = img.shape[1]
  else: 
    num_y = GetNPArrayViewFromSITK(img, has_component_data).shape[1]
  
  w = widg.IntSlider(value=slc_no,
                     min=0,
                     max=num_y,
                     step=1,
                     description='Slice Position:')

  widg.interactive(next_y, slc=w)



def show_tensor_zslice(img, slc_no, scale=1, title=None, margin=0.05, dpi=80, has_component_data=False):
  def next_z(slc):
    show_tensor_slice(img, slc, 2, scale, title, margin, dpi, has_component_data)

  if type(img) == np.ndarray:
    num_z = img.shape[2]
  else: 
    num_z = GetNPArrayViewFromSITK(img, has_component_data).shape[2]
  
  w = widg.IntSlider(value=slc_no,
                     min=0,
                     max=num_z,
                     step=1,
                     description='Slice Position:')

  widg.interactive(next_z, slc=w)

    
# from https://stackoverflow.com/questions/41597177/get-aspect-ratio-of-axes
def get_aspect(ax):
 from operator import sub
 # Total figure size
 figW, figH = ax.get_figure().get_size_inches()
 # Axis size on figure
 _, _, w, h = ax.get_position().bounds
 # Ratio of display units
 disp_ratio = (figH * h) / (figW * w)
 # Ratio of data units
 # Negative over negative because of the order of subtraction
 data_ratio = sub(*ax.get_ylim()) / sub(*ax.get_xlim())

 return disp_ratio / data_ratio

def get_data_ratio(ax):
 # Ratio of data units
 # Negative over negative because of the order of subtraction
 data_ratio = sub(*ax.get_ylim()) / sub(*ax.get_xlim())

 return data_ratio

def show_3d(img, xslices=[], yslices=[], zslices=[], title=None, margin=0.05, dpi=80,
            has_component_data=False):
 size = img.GetSize()
 img_xslices = [img[s,:,:] for s in xslices]
 img_yslices = [img[:,s,:] for s in yslices]
 img_zslices = [img[:,:,s] for s in zslices]
 
 maxlen = max(len(img_xslices), len(img_yslices), len(img_zslices))
 
     
 img_null = sitk.Image([0,0], img.GetPixelID(), img.GetNumberOfComponentsPerPixel())
 
 img_slices = []
 d = 0
 
 if len(img_xslices):
   img_slices += img_xslices + [img_null]*(maxlen-len(img_xslices))
   d += 1
     
 if len(img_yslices):
   img_slices += img_yslices + [img_null]*(maxlen-len(img_yslices))
   d += 1
  
 if len(img_zslices):
   img_slices += img_zslices + [img_null]*(maxlen-len(img_zslices))
   d +=1
 
 if maxlen != 0:
   if img.GetNumberOfComponentsPerPixel() == 1:
     img = sitk.Tile(img_slices, [maxlen,d])
     #TO DO check in code to get Tile Filter working with vector images
   else:
     img_comps = []
     for i in range(0,img.GetNumberOfComponentsPerPixel()):
       img_slices_c = [sitk.VectorIndexSelectionCast(s, i) for s in img_slices]
       img_comps.append(sitk.Tile(img_slices_c, [maxlen,d]))
     img = sitk.Compose(img_comps)
         
 
 show_2d(img, title, margin, dpi)

###########################################
# Simple ITK segmentation overlay methods # 
###########################################

def overlay_contours(img, seg, title):
 img_255 = sitk.Cast(sitk.RescaleIntensity(img), sitk.sitkUInt8)
 seg_255 = sitk.Cast(sitk.RescaleIntensity(seg), sitk.sitkUInt8)
 show_2d(sitk.LabelOverlay(img_255, sitk.LabelContour(seg_255), 1.0), title=title)

def overlay_seg(img, seg, title):
 img_255 = sitk.Cast(sitk.RescaleIntensity(img), sitk.sitkUInt8)
 seg_255 = sitk.Cast(sitk.RescaleIntensity(seg), sitk.sitkUInt8)
 show_2d(sitk.LabelOverlay(img_255, seg_255, 1.0), title=title, has_component_data=True)

def overlay_seg_3d(img, seg, xslices, yslices, zslices, title):
 img_255 = sitk.Cast(sitk.RescaleIntensity(img), sitk.sitkUInt8)
 seg_255 = sitk.Cast(sitk.RescaleIntensity(seg), sitk.sitkUInt8)
 show_3d(sitk.LabelOverlay(img_255, seg_255, 1.0), xslices, yslices, zslices, title=title, has_component_data=True)

#########################
# Histogram Vis Methods #
#########################
def disp_hist_2D(hist, clusterNum, numOfClusters, binSize, title='', margin=0.05, dpi=80):
 # assuming getting a histogram packed into 1-D
 #clusterhist = hist[clusterNum::numOfClusters]
 #binSize = int(math.floor(math.sqrt(clusterhist.shape[0])))
 #clusterhist.reshape(binSize, binSize)
 # Make a figure big enough to accommodate an axis of xpixels by ypixels
 # as well as the ticklabels, etc...
 #if binSize > dpi:
 #  figsize = (1 + margin) * binSize / dpi, (1 + margin) * binSize / dpi
 #else:
 #  figsize = (1 + margin) * dpi / binSize, (1 + margin) * dpi / binSize
 #print("fig size: ", figsize)

 #fig = plt.figure(figsize=figsize, dpi=dpi)
 fig = plt.figure()
 # Make the axis the right size...
 ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])
 
 extent = (0, binSize, binSize, 0)
 
 t = ax.imshow(hist[clusterNum::numOfClusters].reshape(binSize, binSize),extent=extent,interpolation=None)
 
 #if hist.ndim == 2:
 #  t.set_cmap("gray")
 
 if(title):
   plt.title(title)

def disp_hist_3D(hist, clusterNum, numOfClusters, binSize, slcdim, slc, title='', margin=0.05, dpi=80):
 # assuming getting a histogram packed into 1-D

 fig = plt.figure()
 # Make the axis the right size...
 ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])
 
 extent = (0, binSize, binSize, 0)
 
 if slcdim == 0:
   t = ax.imshow(hist[clusterNum::numOfClusters].reshape(binSize, binSize,binSize)[slc,:,:],extent=extent,interpolation=None)
 elif slcdim == 1:
   t = ax.imshow(hist[clusterNum::numOfClusters].reshape(binSize, binSize,binSize)[:,slc,:],extent=extent,interpolation=None)
 else:
   t = ax.imshow(hist[clusterNum::numOfClusters].reshape(binSize, binSize,binSize)[:,:,slc],extent=extent,interpolation=None)
 
 if(title):
   plt.title(title)

########################################
# numpy displacement field vis methods #
########################################

def subtitle(add_title):
 # create longer title from existing axis title
 return(plt.gca().get_title() + " " + add_title)

def plot_grid_2d(hfield, title='', fig=None, margin=0.05, dpi=80):
  
  xsize = hfield.shape[1]
  ysize = hfield.shape[2]

  if xsize > dpi and ysize > dpi:
    figsize = (1 + margin) * ysize / dpi, (1 + margin) * xsize / dpi
  else:
    figsize = (1 + margin) * dpi / ysize, (1 + margin) * dpi / xsize
 
  if fig is not None:
    fg = plt.figure(fig.number)
    full_title = subtitle(title)
  else:
    fg = plt.figure()
    full_title = title
    # Make the axis the right size...
    ax = fg.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])
    
  plt.plot(hfield[0,:,:], hfield[1,:,:], 'b')
  plt.plot(np.transpose(hfield[0,:,:]), np.transpose(hfield[1,:,:]), 'b')

  if(full_title):
    plt.title(full_title)

  ax = plt.gca()
  ax.set_xlim([0,xsize])
  ax.set_ylim([0,ysize])
  

  plt.show()

def quiver_par_curv(curv_p, curv_n, title="", fig=None):
  if fig is not None:
    fg = plt.figure(fig.number)
    full_title = subtitle(title)
  else:
    fg = plt.figure()
    full_title = title
   
  plt.quiver(curv_p[0], curv_p[1], np.gradient(curv_p[0]), np.gradient(curv_p[1]), angles='xy')
  plt.quiver(curv_n[0], curv_n[1], np.gradient(curv_n[0]), np.gradient(curv_n[1]), angles='xy')
  if(full_title):
    plt.title(full_title)

  plt.show()

################
# vis_ellipses #
################
def vis_ellipses(tensor_field, title, points1_x, points1_y, points2_x, points2_y,
                 points3_x, points3_y, points4_x, points4_y, points5_x, points5_y,
                 points6_x, points6_y, save_file=False, filename=''):
  eps11 = tensor_field[0, :, :]
  eps12 = tensor_field[1, :, :]
  eps22 = tensor_field[2, :, :]
  # visualizing ellipses
  ells = []
  tens = np.zeros((2, 2))
  scale = 0.3
  for x in range(1, eps11.shape[0] - 1):
    for y in range(1, eps11.shape[1] - 1):
      # evals and evecs by numpy
      tens[0, 0] = eps11[x, y]
      tens[0, 1] = eps12[x, y]
      tens[1, 0] = eps12[x, y]
      tens[1, 1] = eps22[x, y]
      evals, evecs = np.linalg.eigh(tens)
      angles = np.degrees(np.math.atan2(evecs[1][1], evecs[1][0]))
      ells.append(Ellipse(xy=(x, y), width=scale * evals[1], height=scale * evals[0], angle=angles))
  # plt.figure(1)
  fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'},figsize=(8,8))
  for e in ells:
    ax.add_artist(e)
    e.set_clip_box(ax.bbox)
    e.set_alpha(1)
    e.set_facecolor([0, 0, 0])
  ax.set_xlim(0, eps11.shape[0])
  ax.set_ylim(0, eps11.shape[1])
  ax.set_title(title)
  ax.scatter(points1_x, points1_y, c='r', s=20, alpha=1, label='Original(Geodesic)')
  ax.scatter(points2_x, points2_y, c='k', s=20, alpha=1, label='Analytic(Geodesic)')
  ax.scatter(points3_x, points3_y, c='g', s=20, alpha=1, label='EuclideanInitial(Geodesic)')
  ax.scatter(points4_x, points4_y, c='purple', s=20, alpha=1, label='GMRES(Geodesic)')
  ax.scatter(points5_x, points5_y, c='b', s=20, alpha=1, label='GMRESOrigScaled(Geodesic)')
  ax.scatter(points6_x, points6_y, c='y', s=5, alpha=1, label='Original(Euler)')
  ax.legend()
  if save_file:
    fig.savefig(filename)
    plt.close(fig)
  else:
    plt.show()

def vis_tensors(tensor_field, title, save_file=False, filename='', mask=None,scale=0.3, opacity=0.5, show_axis_labels=True, ax=None,zorder=1,stride=None):
  eps11 = tensor_field[0, :, :]
  eps12 = tensor_field[1, :, :]
  eps22 = tensor_field[2, :, :]
  # visualizing ellipses
  ells = []
  tens = np.zeros((2, 2))
  #scale = 0.3

  if stride is None:
    stride = 1

  tens = np.zeros((eps11.shape[0],eps11.shape[1],2,2))
  if mask is None:
    tens[:,:,0,0] = eps11[:,:]
    tens[:,:,0,1] = eps12[:,:]
    tens[:,:,1,0] = eps12[:,:]
    tens[:,:,1,1] = eps22[:,:]
  else:
    tens[:,:,0,0] = mask*eps11[:,:]
    tens[:,:,0,1] = mask*eps12[:,:]
    tens[:,:,1,0] = mask*eps12[:,:]
    tens[:,:,1,1] = mask*eps22[:,:]
   
  evals, evecs = np.linalg.eigh(tens)

  angles = np.degrees(np.arctan2(evecs[:,:,1,1],evecs[:,:,1,0]))
 
  scaled_evals = scale * evals

  if mask is None:
    ells = [Ellipse((x,y), width=scaled_evals[x,y,1], height = scaled_evals[x,y,0], angle=angles[x,y],zorder=zorder) for y in range(eps11.shape[1]) for x in range(eps11.shape[0])]
  else:
    for x in range(0,eps11.shape[0],stride):
      for y in range(0,eps11.shape[1],stride):
       if mask[x,y]:
         ells.append(Ellipse((x,y), width=scaled_evals[x,y,1], height = scaled_evals[x,y,0], angle=angles[x,y],zorder=zorder))
  
  #for x in range(1, eps11.shape[0] - 1):
  #  for y in range(1, eps11.shape[1] - 1):
  #    # evals and evecs by numpy
  #    tens[0, 0] = eps11[x, y]
  #    tens[0, 1] = eps12[x, y]
  #    tens[1, 0] = eps12[x, y]
  #    tens[1, 1] = eps22[x, y]
  #    evals, evecs = np.linalg.eigh(tens)
  #    angles = np.degrees(np.math.atan2(evecs[1][1], evecs[1][0]))
  #    ells.append(Ellipse(xy=(x, y), width=scale * evals[1], height=scale * evals[0], angle=angles))
  # plt.figure(1)
  if ax is None:
    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'},figsize=(8,8))
  else:
    fig = plt.gcf()
  for e in ells:
    ax.add_artist(e)
    e.set_clip_box(ax.bbox)
    e.set_alpha(opacity)
    e.set_facecolor([0, 0, 0])
  ax.set_xlim(0, eps11.shape[0])
  ax.set_ylim(0, eps11.shape[1])
#   if not show_axis_labels:
#     plt.xticks([])
#     plt.yticks([])
  plt.axis('off')
#   ax.set_title(title)
  return(fig)

# def vis_path(points_x, points_y, fig=None, label='', color='b', sz=20, alpha=1, save_file=False, filename='', yscale='linear', ax=None,zorder=1,stride=None):
#   if ax is None:
#     if fig is None:
#       plt.figure()
#     else:
#       plt.figure(fig.number)
#     ax = plt.gca()
#   if stride is None:
#     stride=1
#   ax.scatter(points_x[::stride], points_y[::stride], c=color, s=sz, alpha=alpha, label=label,zorder=zorder)
#   plt.yscale(yscale)
#   ax.legend()
#   if save_file:
#     fig.savefig(filename)
#     plt.close(fig)
#   else:
#     plt.show()
#   return(fig)

def vis_path(points_x, points_y, fig=None, label='', color='b', sz=20, alpha=1, save_file=False, filename='', yscale='linear', ax=None,zorder=1,stride=None,show_legend=True):
  if ax is None:
    ax = plt.gca()
  if stride is None:
    stride=1
  ax.scatter(points_x[::stride], points_y[::stride], c=color, s=sz, alpha=alpha, label=label,zorder=zorder)
  plt.yscale(yscale)
  if show_legend:
    ax.legend(loc='lower left',fontsize=13.5)
  if save_file:
    fig.savefig(filename)
    plt.close(fig)
  else:
    plt.show()
  return(fig)

def gen_and_vis_paths(file_pattern, start_iter, stop_iter, inc_iter, start_coords, interp_colors, geo_iters, geo_delta_t, 
                      zoom_box=None, title='', tens_scale=None, do_inverse=False, legend_pattern='',
                      line_width=40, final_line_width=10):

  #geo_line_width=40
  #eul_line_width=10
  show_axis_labels=False
  if tens_scale is None:
    tens_scale = 1000
  if not legend_pattern:
    legend_pattern = "Atlas Geodesic Iter {}"
  init_velocities = [None]

  iso_tens=np.zeros((2,2))
  iso_tens[0,0]=.0001
  iso_tens[1,1]=.0001

  final_tens = ReadTensors(file_pattern.format(stop_iter))
  xsz = final_tens.shape[0]
  ysz = final_tens.shape[1]
  final_mask = np.zeros((xsz, ysz)) 
  final_mask[final_tens[:,:,0]>0.0002] = 1
  if do_inverse:
    tens_full = np.zeros((xsz,ysz,2,2))
    tens_full[:,:,0,0] = final_tens[:,:,0]
    tens_full[:,:,0,1] = final_tens[:,:,1]
    tens_full[:,:,1,0] = final_tens[:,:,1]
    tens_full[:,:,1,1] = final_tens[:,:,2]
    inv_tens = np.linalg.inv(tens_full)
    inv_tens[final_mask==0] = iso_tens
    final_tens_4_path = np.zeros((3,xsz,ysz))
    final_tens_4_path[0,:,:] = inv_tens[:,:,0,0]
    final_tens_4_path[1,:,:] = inv_tens[:,:,0,1]
    final_tens_4_path[2,:,:] = inv_tens[:,:,1,1]
  else:
    final_tens_4_path = np.transpose(final_tens,(2,0,1))
    
  finalgeox, finalgeoy = geo.geodesicpath(final_tens_4_path, final_mask,\
                              start_coords[0], init_velocities[0], \
                              geo_delta_t, iter_num=geo_iters, both_directions=True)

  tens_fig = vis_tensors(final_tens_4_path, title, False,scale=tens_scale)

  idx = 0
  for it in range(start_iter, stop_iter, inc_iter):
    tens = ReadTensors(file_pattern.format(it))
    mask = np.zeros((xsz,ysz)) 
    mask[tens[:,:,0]>0.0002] = 1

    if do_inverse:
      tens_full = np.zeros((xsz,ysz,2,2))
      tens_full[:,:,0,0] = tens[:,:,0]
      tens_full[:,:,0,1] = tens[:,:,1]
      tens_full[:,:,1,0] = tens[:,:,1]
      tens_full[:,:,1,1] = tens[:,:,2]
      inv_tens = np.linalg.inv(tens_full)
      inv_tens[mask==0] = iso_tens
      tens_4_path = np.zeros((3,xsz,ysz))
      tens_4_path[0,:,:] = inv_tens[:,:,0,0]
      tens_4_path[1,:,:] = inv_tens[:,:,0,1]
      tens_4_path[2,:,:] = inv_tens[:,:,1,1]
    else:
      tens_4_path = np.transpose(tens,(2,0,1))

    geox, geoy = geo.geodesicpath(tens_4_path, mask,\
                              start_coords[0], init_velocities[0], \
                              geo_delta_t, iter_num=geo_iters, both_directions=True)

    #vis_path(geox, geoy, tens_fig, f"Atlas Geodesic iter: {it}", interp_colors[idx], geo_line_width, 1, False)
    vis_path(geox, geoy, tens_fig, legend_pattern.format(it), interp_colors[idx], line_width, 1, False)

    idx += 1
    if idx >= len(interp_colors):
      idx = 0

  #vis_path(finalgeox, finalgeoy, tens_fig, f"Atlas Geodesic iter: {stop_iter}", interp_colors[idx], geo_line_width, 1, False)
  vis_path(finalgeox, finalgeoy, tens_fig, legend_pattern.format(stop_iter), 'k', final_line_width, 1, False)
    
  if not show_axis_labels:
    plt.xticks([])
    plt.yticks([])

  if zoom_box:
    plt.xlim(zoom_box[0])
    plt.ylim(zoom_box[1])

    
def gen_and_vis_paths_per_coords(tens, start_coords, interp_colors, geo_iters, geo_delta_t,
                                 zoom_box=None, title='',tens_scale=None,do_inverse=False,
                                 legend_pattern='', line_width=40):

  geo_line_width=40
  eul_line_width=10
  show_axis_labels=False
  if tens_scale is None:
    tens_scale = 1000
  if not legend_pattern:
    legend_pattern = "Atlas Geodesic Through {}"
  init_velocities = [None]

  iso_tens=np.zeros((2,2))
  iso_tens[0,0]=.0001
  iso_tens[1,1]=.0001

  xsz = tens.shape[0]
  ysz = tens.shape[1]
    
  if do_inverse:
    if len(tens.shape) == 4:
      inv_tens = np.linalg.inv(tens)
      mask = np.zeros((xsz,ysz)) 
      mask[tens[:,:,0,0]>0.0002] = 1  
    else:
      tens_full = np.zeros((xsz,ysz,2,2))
      tens_full[:,:,0,0] = tens[:,:,0]
      tens_full[:,:,0,1] = tens[:,:,1]
      tens_full[:,:,1,0] = tens[:,:,1]
      tens_full[:,:,1,1] = tens[:,:,2]
      mask = np.zeros((xsz,ysz)) 
      mask[tens_full[:,:,0,0]>0.0002] = 1 
      inv_tens = np.linalg.inv(tens_full)
    inv_tens[mask==0] = iso_tens
    tens_4_path = np.zeros((3,xsz,ysz))
    tens_4_path[0,:,:] = inv_tens[:,:,0,0]
    tens_4_path[1,:,:] = inv_tens[:,:,0,1]
    tens_4_path[2,:,:] = inv_tens[:,:,1,1]
  else:
    if (len(tens.shape)==3):
      tens_4_path = np.transpose(tens,(2,0,1))
    else:
      tens_4_path = np.zeros((3,xsz,ysz))
      tens_4_path[0,:,:] = tens[:,:,0,0]
      tens_4_path[1,:,:] = tens[:,:,0,1]
      tens_4_path[2,:,:] = tens[:,:,1,1]
    mask = np.zeros((xsz,ysz)) 
    mask[tens_4_path[0,:,:]>0.0002] = 1 
   
  tens_fig = vis_tensors(tens_4_path, title, False,scale=tens_scale)

  idx = 0
  for it in range(len(start_coords)):
    geox, geoy = geo.geodesicpath(tens_4_path, mask,\
                              start_coords[it], init_velocities[0], \
                              geo_delta_t, iter_num=geo_iters, both_directions=True)

#    vis_path(geox, geoy, tens_fig, f"Atlas Geodesic {start_coords[it]}", interp_colors[idx], geo_line_width, 1, False)
    vis_path(geox, geoy, tens_fig, legend_pattern.format(start_coords[it]), interp_colors[idx], line_width, 1, False)

    idx += 1
    if idx >= len(interp_colors):
      idx = 0

  if not show_axis_labels:
    plt.xticks([])
    plt.yticks([])

  if zoom_box:
    plt.xlim(zoom_box[0])
    plt.ylim(zoom_box[1])
    

 
####################################
# read data, plot and save to file #
####################################
def disp_scalar_to_file(datafile, figfile, title='', threshold=None):
    img = readRaw(datafile)
    img = np.reshape(img, (100,100)).T
    if threshold is not None:
        img[img > threshold[1]] = threshold[1]
        img[img < threshold[0]] = threshold[0]   
    show_2d(img, title)
    plt.savefig(figfile,bbox_inches='tight')
def disp_vector_to_file(datafile, figfiles, title='', threshold=None):
    img = readRaw(datafile)
    img = np.transpose(np.reshape(img, (100,100,2)), (1,0,2))
    if threshold is not None:
        img[img > threshold[1]] = threshold[1]
        img[img < threshold[0]] = threshold[0]   
    show_2d(img[:,:,0], title)
    plt.savefig(figfiles[0])  
    show_2d(img[:,:,1], title)
    plt.savefig(figfiles[1]) 
def disp_tensor_to_file(datafile, figfiles, title='', threshold=None):
    img = readRaw(datafile)
    img = np.transpose(np.reshape(img, (100,100,3)), (1,0,2))
    if threshold is not None:
        img[img > threshold[1]] = threshold[1]
        img[img < threshold[0]] = threshold[0]   
    show_2d(img[:,:,0], title)
    plt.savefig(figfiles[0])  
    show_2d(img[:,:,1], title)
    plt.savefig(figfiles[1])  
    show_2d(img[:,:,2], title)
    plt.savefig(figfiles[2])
def disp_gradG_to_file(datafile, figfiles, title='', threshold=None):
    img = readRaw(datafile)
    img = np.transpose(np.reshape(img, (2,100,100,3)), (2,1,3,0))
    if threshold is not None:
        img[img > threshold[1]] = threshold[1]
        img[img < threshold[0]] = threshold[0]   
    show_2d(img[:,:,0,0], title)
    plt.savefig(figfiles[0])  
    show_2d(img[:,:,1,0], title)
    plt.savefig(figfiles[1])  
    show_2d(img[:,:,2,0], title)
    plt.savefig(figfiles[2])
    show_2d(img[:,:,0,1], title)
    plt.savefig(figfiles[3])  
    show_2d(img[:,:,1,1], title)
    plt.savefig(figfiles[4])  
    show_2d(img[:,:,2,1], title)
    plt.savefig(figfiles[5])
def disp_gradA_to_file(datafile, figfiles, title='', threshold=None):
    img = readRaw(datafile)
    img = np.transpose(np.reshape(img, (2,100,100,2)), (2,1,3,0))
    show_2d(img[:,:,0,0], title)
    plt.savefig(figfiles[0])  
    show_2d(img[:,:,1,0], title)
    plt.savefig(figfiles[1])  
    show_2d(img[:,:,0,1], title)
    plt.savefig(figfiles[2])  
    show_2d(img[:,:,1,1], title)
    plt.savefig(figfiles[3])
