# Do lazy importing here as suggested at https://pypi.org/project/lazy-import/
#
# Note that it is okay to have something like
# from lazy_imports import np
# at the top of each file.  The import will not actually happen
# until you try and use np within a function or class.

import lazy_import

# TODO standardize on naming convention here
np = lazy_import.lazy_module("numpy")
ndimage = lazy_import.lazy_module("scipy.ndimage")
linalg = lazy_import.lazy_module("scipy.linalg")
sio = lazy_import.lazy_module("scipy.io")
loadmat = lazy_import.lazy_callable("scipy.io.loadmat")
savemat = lazy_import.lazy_callable("scipy.io.savemat")
sct = lazy_import.lazy_module("scipy.stats")
griddata = lazy_import.lazy_callable("scipy.interpolate.griddata")
interp2d = lazy_import.lazy_callable("scipy.interpolate.interp2d")
splprep = lazy_import.lazy_callable("scipy.interpolate.splprep")
splev = lazy_import.lazy_callable("scipy.interpolate.splev")
LinearOperator = lazy_import.lazy_callable("scipy.sparse.linalg.LinearOperator")
gmres = lazy_import.lazy_callable("scipy.sparse.linalg.gmres")

# WARNING! from lazy_imports import itk does not work well, especially with itkwidgets.view
# FOR NOW, be sure to import itk directly, prior to lazy_import of np, itkwidgets, etc.
# Otherwise will see strange errors such as itk.PointSet or itk.Image not existing
# itk = lazy_import.lazy_module("itk")
itkwidgets = lazy_import.lazy_module("itkwidgets")
itkview = lazy_import.lazy_callable("itkwidgets.view")
pv = lazy_import.lazy_module("pyvista")
sitk = lazy_import.lazy_module("SimpleITK")
nib = lazy_import.lazy_module("nibabel")
nils = lazy_import.lazy_module("nilearn.surface")
nilp = lazy_import.lazy_module("nilearn.plotting")

nx = lazy_import.lazy_module("networkx")
manifold = lazy_import.lazy_module("sklearn.manifold")
svm = lazy_import.lazy_module("sklearn.svm")
make_pipeline = lazy_import.lazy_module("sklearn.pipeline.make_pipeline")
pd = lazy_import.lazy_module("pandas")
torch = lazy_import.lazy_module("torch")

axes3d = lazy_import.lazy_module("mpl_toolkits.mplot3d.axes3d")
art3d = lazy_import.lazy_module("mpl_toolkits.mplot3d.art3d")
plt = lazy_import.lazy_module("matplotlib.pyplot")
cm = lazy_import.lazy_module("matplotlib.cm")
PatchCollection = lazy_import.lazy_callable("matplotlib.collections.PatchCollection")
EllipseCollection = lazy_import.lazy_callable("matplotlib.collections.EllipseCollection")
Ellipse = lazy_import.lazy_callable("matplotlib.patches.Ellipse")
ipywidgets = lazy_import.lazy_module("ipywidgets")
interactive = lazy_import.lazy_callable("ipywidgets.interactive")

