# test basics of SITK image vis
from .. import data
from .. import disp
from ..data import constants as dc

def runTest():
  tst_img = data.gen.newImage(dc.IM_TYPE_SITK, [26, 26, 26], dc.DATA_TYPE_FLOAT)
  disp.vis.show_3d(tst_img,yslices=range(7,9))

  for i in range(5,15):
    for j in range(4,10):
      for k in range(8,20):
        tst_img.SetPixel(i,j,k, i*j*k)

  disp.vis.show_info(tst_img)
  disp.vis.show_slice(tst_img, 9, 0)
  disp.vis.show_3d(tst_img,yslices=range(7,9))

if __name__ == "__main__":
  runTest()
