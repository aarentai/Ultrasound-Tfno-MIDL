# test basics of tensor field generation

from optparse import OptionParser
import random

#from .. import data
#from .. import disp
#from ..data.io import WriteTensorNPArray
import data.gen
import disp
from data.io import WriteTensorNPArray, WriteScalarNPArray

def testAnnulusParametric():
  ann_p, ann_n = data.gen.make_annulus(1, 1 / 5.0, 10)
  disp.vis.quiver_par_curv(ann_p, ann_n)

def testAnnulusTens2D(xsz, ysz, out_prefix):
  circ_rng = (-1.4,1.4)
  img, tens, seed, xrg, yrg = data.gen.gen_2D_annulus(xsz, ysz, 6.0, True, False, False, xrng = circ_rng, yrng = circ_rng)

  WriteTensorNPArray(tens, out_prefix + "tens.nhdr")
  WriteScalarNPArray(img, out_prefix + "mask.nhdr")
  WriteScalarNPArray(seed, out_prefix + "seed.nhdr")

def testCubic1Tens2D(xsz, ysz, out_prefix, xrg=None, yrg=None, zero_padding_width=None):
  a3 = -4 #-4.0
  a2 = 4 #2.0
  a1 = 2 #10.0
  a0 = 0.0 # 0.0
  b3 = 8 #8.0
  b2 = -12 #-10.0
  b1 = 6 #2.0
  b0 = 0 #0
  ##a3 = -4.0
  ##a2 = 2.0
  ##a1 = 10.0
  ##a0 = 0.0
  ##b3 = 8.0
  ##b2 = -10.0
  ##b1 = 2.0
  ##b0 = 0
  #c1 = lambda t: data.gen.cubic(t, a3, a2, a1, a0)
  #dc1 = lambda t, dt: data.gen.d_cubic(t, a3, a2, a1, a0, dt)
  #c2 = lambda t: data.gen.cubic(t, b3, b2, b1, b0)
  #dc2 = lambda t, dt: data.gen.d_cubic(t, b3, b2, b1, b0, dt)
  ## TODO ok if xrg, yrg not isotropic?
  #(cubic_no_blur_img, cubic_no_blur_ten, cubic_no_blur_seed, cubic_no_blur_xrg, cubic_no_blur_yrg) = data.gen.gen_2D_tensor_image(xsz, ysz, 0, 1, 1000, c1, dc1, c2, dc2, 1/5.0, 15, 6.0, 0.05, 0.95,True,False,False, xrg, yrg, zero_padding_width=zero_padding_width)

  #WriteTensorNPArray(cubic_no_blur_ten, out_prefix + "tens.nhdr")
  #WriteScalarNPArray(cubic_no_blur_img, out_prefix + "mask.nhdr")
  #WriteScalarNPArray(cubic_no_blur_seed, out_prefix + "seed.nhdr")
  #return (cubic_no_blur_xrg, cubic_no_blur_yrg)

  return(testGenAndWriteCubic(xsz, ysz, a0, a1, a2, a3, b0, b1, b2, b3, out_prefix, xrg, yrg, zero_padding_width))

def testCubic2Tens2D(xsz, ysz, out_prefix, xrg = None, yrg = None, zero_padding_width=None):
  a3_2 = -3 #-4 #-4.0
  a2_2 = 3.5 #4 #2.0
  a1_2 = 1.5 #2 #10.0
  a0_2 = 0.0 #0.0 # 0.0
  b3_2 = 6.5 #8 #8.0
  b2_2 = -11 #-12 #-10.0
  b1_2 = 6.5 #6 #2.0
  b0_2 = 0 #0 #0
  ##a3_2 = -3.0
  ##a2_2 = 1.5
  ##a1_2 = 9.5
  ##a0_2 = 0.0
  ##b3_2 = 6.5
  ##b2_2 = -9
  ##b1_2 = 2.5
  ##b0_2 = 0
  #c1_2 = lambda t: data.gen.cubic(t, a3_2, a2_2, a1_2, a0_2)
  #dc1_2 = lambda t, dt: data.gen.d_cubic(t, a3_2, a2_2, a1_2, a0_2, dt)
  #c2_2 = lambda t: data.gen.cubic(t, b3_2, b2_2, b1_2, b0_2)
  #dc2_2 = lambda t, dt: data.gen.d_cubic(t, b3_2, b2_2, b1_2, b0_2, dt)
  #(cubic_no_blur_img2, cubic_no_blur_ten2, cubic_no_blur_seed2, cubic_no_blur_xrg2, cubic_no_blur_yrg2) = data.gen.gen_2D_tensor_image(xsz, ysz, 0, 1, 1000, c1_2, dc1_2, c2_2, dc2_2, 1/5.0, 15, 6.0, 0.05, 0.95,True,False,False, xrg, yrg, zero_padding_width=zero_padding_width)

  #WriteTensorNPArray(cubic_no_blur_ten2, out_prefix + "tens.nhdr")
  #WriteScalarNPArray(cubic_no_blur_img2, out_prefix + "mask.nhdr")
  #WriteScalarNPArray(cubic_no_blur_seed2, out_prefix + "seed.nhdr")

  return(testGenAndWriteCubic(xsz, ysz, a0_2, a1_2, a2_2, a3_2, b0_2, b1_2, b2_2, b3_2, out_prefix, xrg, yrg, zero_padding_width))

def testGenAndWriteCubic(xsz, ysz, a0, a1, a2, a3, b0, b1, b2, b3, out_prefix, xrg=None, yrg=None, zero_padding_width=None):
  c1 = lambda t: data.gen.cubic(t, a3, a2, a1, a0)
  dc1 = lambda t, dt: data.gen.d_cubic(t, a3, a2, a1, a0, dt)
  c2 = lambda t: data.gen.cubic(t, b3, b2, b1, b0)
  dc2 = lambda t, dt: data.gen.d_cubic(t, b3, b2, b1, b0, dt)
  # TODO ok if xrg, yrg not isotropic?
  (cubic_no_blur_img, cubic_no_blur_ten, cubic_no_blur_seed, cubic_no_blur_xrg, cubic_no_blur_yrg) = data.gen.gen_2D_tensor_image(xsz, ysz, 0, 1, 1000, c1, dc1, c2, dc2, 1/5.0, 15, 6.0, 0.05, 0.95,True,False,False, xrg, yrg, zero_padding_width=zero_padding_width)

  WriteTensorNPArray(cubic_no_blur_ten, out_prefix + "tens.nhdr")
  WriteScalarNPArray(cubic_no_blur_img, out_prefix + "mask.nhdr")
  WriteScalarNPArray(cubic_no_blur_seed, out_prefix + "seed.nhdr")
  return (cubic_no_blur_xrg, cubic_no_blur_yrg)

def testRandomCubicTens2D(xsz, ysz, out_prefix, num_cubics=None, xrg=None, yrg=None, zero_padding_width=None):
  a3 = -4 #-4.0
  a2 = 4 #2.0
  a1 = 2 #10.0
  a0 = 0.0 # 0.0
  b3 = 8 #8.0
  b2 = -12 #-10.0
  b1 = 6 #2.0
  b0 = 0 #0

  a3_2 = -3 #-4 #-4.0
  a2_2 = 3.5 #4 #2.0
  a1_2 = 1.5 #2 #10.0
  a0_2 = 0.0 #0.0 # 0.0
  b3_2 = 6.5 #8 #8.0
  b2_2 = -11 #-12 #-10.0
  b1_2 = 6.5 #6 #2.0
  b0_2 = 0 #0 #0

  if num_cubics is None:
    num_cubics = 2
  if num_cubics >= 1:
    xrg, yrg = testGenAndWriteCubic(xsz, ysz, a0, a1, a2, a3, b0, b1, b2, b3, out_prefix + "cubic1_", xrg, yrg, zero_padding_width)
  else:
    return

  if num_cubics >= 2:
    testGenAndWriteCubic(xsz, ysz, a0_2, a1_2, a2_2, a3_2, b0_2, b1_2, b2_2, b3_2, out_prefix + "cubic2_", xrg, yrg, zero_padding_width)
  else:
    return

  #random.seed(86) # for repeated random.uniform
  #rfs = [1, 0]
  #random.seed(867530)
  #for cc in range(3,num_cubics+1):
  #  rfs.append(random.uniform(0,1))
  rfs = [1, 0, 0.2, 0.4, 0.5, 0.6, 0.8]
  
  for cc in range(3,num_cubics+1):
    rf = rfs[cc-1]
    print("random percent",rf)
    a3_r = a3_2 + rf * (a3 - a3_2)
    a2_r = a2_2 + rf * (a2 - a2_2)
    a1_r = a1_2 + rf * (a1 - a1_2)
    a0_r = 0.0
    b3_r = b3_2 + rf * (b3 - b3_2)
    b2_r = b2_2 + rf * (b2 - b2_2)
    b1_r = b1_2 + rf * (b1 - b1_2)
    b0_r = 0.0
    #a3_r = random.uniform(a3, a3_2)
    #a2_r = random.uniform(a2_2, a2)
    #a1_r = random.uniform(a1_2, a1)
    #a0_r = 0.0
    #b3_r = random.uniform(b3_2, b3)
    #b2_r = random.uniform(b2, b2_2)
    #b1_r = random.uniform(b1, b1_2)
    #b0_r = 0.0
    testGenAndWriteCubic(xsz, ysz, a0_r, a1_r, a2_r, a3_r, b0_r, b1_r, b2_r, b3_r, out_prefix + f"cubic{cc}_" , xrg, yrg, zero_padding_width)
  return(xrg, yrg)

def testVaryRectangleTens2D(xsz, ysz, out_prefix):
  tens = data.gen.gen_2D_rectangle_gradient_ratio(xsz, ysz, 1/6.0, rotation = 0, do_isotropic=True, zero_padding_width=0)
  WriteTensorNPArray(tens, out_prefix + "rect_vary_tens.nhdr")

def testIsoRectangleTens2D(xsz, ysz, out_prefix):
  tens = data.gen.gen_2D_rectangle_constant_ratio(xsz, ysz, 1, rotation = 0, do_isotropic=True, zero_padding_width=0)
  WriteTensorNPArray(tens, out_prefix + "rect_iso_tens.nhdr")

  
if __name__ == "__main__":
  usage = """
%prog [options]

generates an annulus and two different cubic tensor fields.  
"""
  optparser = OptionParser(usage=usage)
  optparser.add_option("-x", "--xsize", dest="xsz", default="100",
                       help="Size of output image in x direction.")
  optparser.add_option("-y", "--ysize", dest="ysz", default="100",
                       help="Size of output image in y direction.")
  optparser.add_option("-z", "--zsize", dest="zsz", default="",
                       help="Size of output image in z direction, if not provided a 2D image is created.")
  optparser.add_option("-o", "--outdir", dest="outdir", default=".",
                       help="Directory to which resulting images will be written.")
  optparser.add_option("-n", "--numcubics", dest="numcubics", default="2",
                       help="Number of cubic functions to generate")
  
  (options, args) = optparser.parse_args()

  do_3d = False
  if options.zsz:
    do_3d = True

  if do_3d:
    print("3D generation not implemented yet")
  else:
    testAnnulusTens2D(int(options.xsz), int(options.ysz), options.outdir + "/metpy_annulus_")
    cubic_xrg = [-0.4, 2.6]
    cubic_yrg =[-0.4, 2.6]
    cubic_xrg = [-1.25, 8.4]
    cubic_yrg =[-1.25, 8.4]
    cubic_xrg = None
    cubic_yrg = None

    testRandomCubicTens2D(int(options.xsz), int(options.ysz), options.outdir + "/metpy_", int(options.numcubics), cubic_xrg, cubic_yrg, 4)
    #xrg, yrg = testCubic1Tens2D(int(options.xsz), int(options.ysz), options.outdir + "/metpy_cubic1_", cubic_xrg, cubic_yrg,4)
    #testCubic2Tens2D(int(options.xsz), int(options.ysz), options.outdir + "/metpy_cubic2_", xrg, yrg,4)

    testVaryRectangleTens2D(10, 5, options.outdir + "/metpy_")
    testIsoRectangleTens2D(10, 5, options.outdir + "/metpy_")
