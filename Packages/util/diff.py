from lazy_imports import np, torch
from util import maskops as mo
from data.convert import get_framework


def get_first_order_derivative(tensor, direction, accuracy):
    fw, fw_name = get_framework(tensor)
    
    if accuracy==1:
        d_tensor = fw.roll(tensor,-1,direction)-tensor
    elif accuracy==2:
        d_tensor = 0.5*fw.roll(tensor,-1,direction)\
                   -0.5*fw.roll(tensor,1,direction)
    elif accuracy==4:
        d_tensor = -1/12*fw.roll(tensor,-2,direction)\
                    +2/3*fw.roll(tensor,-1,direction)\
                    -2/3*fw.roll(tensor,1,direction)\
                    +1/12*fw.roll(tensor,2,direction)
    elif accuracy==6:
        d_tensor = 1/60*fw.roll(tensor,-3,direction)\
                    -3/20*fw.roll(tensor,-2,direction)\
                    +3/4*fw.roll(tensor,-1,direction)\
                    -3/4*fw.roll(tensor,1,direction)\
                    +3/20*fw.roll(tensor,2,direction)\
                    -1/60*fw.roll(tensor,3,direction)
    elif accuracy==8:
        d_tensor = -1/280*fw.roll(tensor,-4,direction)\
                    +4/105*fw.roll(tensor,-3,direction)\
                    -1/5*fw.roll(tensor,-2,direction)\
                    +4/5*fw.roll(tensor,-1,direction)\
                    -4/5*fw.roll(tensor,1,direction)\
                    +1/5*fw.roll(tensor,2,direction)\
                    -4/105*fw.roll(tensor,3,direction)\
                    +1/280*fw.roll(tensor,4,direction)
    else:
        d_tensor = 0.5*torch.roll(tensor,-1,direction)\
                   -0.5*torch.roll(tensor,1,direction)
    return d_tensor

def select_diffs_3d(btype,zero_shifts=True):
  # choose the correct diff function based on boundary type
  if btype[0:8] == "interior":
    xdiff = gradx_idx_3d
    ydiff = grady_idx_3d
    zdiff = gradz_idx_3d

  else:
    if "shiftx" in btype:
      if zero_shifts:
        xdiff = diff_0
      else:
        if "shiftxtob" in btype:
          xdiff = gradx_shiftbottom_idx_3d
        elif "shiftxtot" in btype:
          xdiff = gradx_shifttop_idx_3d
        elif "shiftxtore" in btype:
          xdiff = gradx_shiftrear_idx_3d
        elif "shiftxtof" in btype:
          xdiff = gradx_shiftfront_idx_3d
        else:
          xdiff = diff_0
    elif "shiftleft" in btype:
      if zero_shifts:
        xdiff = diff_0
      else:
        if "shiftlefttob" in btype:
          xdiff = left_shiftbottom_idx_3d
        elif "shiftlefttot" in btype:
          xdiff = left_shifttop_idx_3d
        elif "shiftlefttore" in btype:
          xdiff = left_shiftrear_idx_3d
        elif "shiftlefttof" in btype:
          xdiff = left_shiftfront_idx_3d
        else:
          xdiff = diff_0
    elif "shiftright" in btype:
      if zero_shifts:
        xdiff = diff_0
      else:
        if "shiftrighttob" in btype:
          xdiff = right_shiftbottom_idx_3d
        elif "shiftrighttot" in btype:
          xdiff = right_shifttop_idx_3d
        elif "shiftrighttore" in btype:
          xdiff = right_shiftrear_idx_3d
        elif "shiftrighttof" in btype:
          xdiff = right_shiftfront_idx_3d
        else:
          xdiff = diff_0
    elif "left" in btype:
      xdiff = left_diff_idx_3d
    elif "right" in btype:
      xdiff = right_diff_idx_3d
    else:
      xdiff = gradx_idx_3d
    if "shifty" in btype:
      if zero_shifts:
        ydiff = diff_0
      else:
        if "shiftytol" in btype:
          ydiff = grady_shiftleft_idx_3d
        elif "shiftytori" in btype:
          ydiff = grady_shiftright_idx_3d
        elif "shiftytore" in btype:
          ydiff = grady_shiftrear_idx_3d
        elif "shiftytof" in btype:
          ydiff = grady_shiftfront_idx_3d
        else:
          ydiff = diff_0
    elif "shiftbottom" in btype:
      if zero_shifts:
        ydiff = diff_0
      else:
        if "shiftbottomtol" in btype:
          ydiff = bottom_shiftleft_idx_3d
        elif "shiftbottomtori" in btype:
          ydiff = bottom_shiftright_idx_3d
        elif "shiftbottomtore" in btype:
          ydiff = bottom_shiftrear_idx_3d
        elif "shiftbottomtof" in btype:
          ydiff = bottom_shiftfront_idx_3d
        else:
          ydiff = diff_0
    elif "shifttop" in btype:
      if zero_shifts:
        ydiff = diff_0
      else:
        if "shifttoptol" in btype:
          ydiff = top_shiftleft_idx_3d
        elif "shifttoptori" in btype:
          ydiff = top_shiftright_idx_3d
        elif "shifttoptore" in btype:
          ydiff = top_shiftrear_idx_3d
        elif "shifttoptof" in btype:
          ydiff = top_shiftfront_idx_3d
        else:
          ydiff = diff_0
    elif "bottom" in btype:
      ydiff = bottom_diff_idx_3d
    elif "top" in btype:
      ydiff = top_diff_idx_3d
    else:
      ydiff = grady_idx_3d
    if "shiftz" in btype:
      if zero_shifts:
        zdiff = diff_0
      else:
        if "shiftztol" in btype:
          zdiff = gradz_shiftleft_idx_3d
        elif "shiftztori" in btype:
          zdiff = gradz_shiftright_idx_3d
        elif "shiftztob" in btype:
          zdiff = gradz_shiftbottom_idx_3d
        elif "shiftztot" in btype:
          zdiff = gradz_shifttop_idx_3d
        else:
          zdiff = diff_0
    elif "shiftrear" in btype:
      if zero_shifts:
        zdiff = diff_0
      else:
        if "shiftreartol" in btype:
          zdiff = rear_shiftleft_idx_3d
        elif "shiftreartori" in btype:
          zdiff = rear_shiftright_idx_3d
        elif "shiftreartob" in btype:
          zdiff = rear_shiftbottom_idx_3d
        elif "shiftreartot" in btype:
          zdiff = rear_shifttop_idx_3d
        else:
          zdiff = diff_0
    elif "shiftfront" in btype:
      if zero_shifts:
        zdiff = diff_0
      else:
        if "shiftfronttol" in btype:
          zdiff = front_shiftleft_idx_3d
        elif "shiftfronttori" in btype:
          zdiff = front_shiftright_idx_3d
        elif "shiftfronttob" in btype:
          zdiff = front_shiftbottom_idx_3d
        elif "shiftfronttot" in btype:
          zdiff = front_shifttop_idx_3d
        else:
          zdiff = diff_0
    elif "rear" in btype:
      zdiff = rear_diff_idx_3d
    elif "front" in btype:
      zdiff = front_diff_idx_3d
    else:
      zdiff = gradz_idx_3d
  return(xdiff, ydiff, zdiff)
# end select_diffs_3d

def select_eigv_diffs_3d(btype):
  # choose the correct diff function based on boundary type
  if btype[0:8] == "interior":
    xdiff = eigv_gradx_idx_3d
    ydiff = eigv_grady_idx_3d
    zdiff = eigv_gradz_idx_3d

  else:
    if "shiftx" in btype:
      if "shiftxtob" in btype:
        xdiff = eigv_gradx_shiftbottom_idx_3d
      elif "shiftxtot" in btype:
        xdiff = eigv_gradx_shifttop_idx_3d
      elif "shiftxtore" in btype:
        xdiff = eigv_gradx_shiftrear_idx_3d
      elif "shiftxtof" in btype:
        xdiff = eigv_gradx_shiftfront_idx_3d
      else:
        xdiff = eigv_diff_0
    elif "shiftleft" in btype:
      if "shiftlefttob" in btype:
        xdiff = eigv_left_shiftbottom_idx_3d
      elif "shiftlefttot" in btype:
        xdiff = eigv_left_shifttop_idx_3d
      elif "shiftlefttore" in btype:
        xdiff = eigv_left_shiftrear_idx_3d
      elif "shiftlefttof" in btype:
        xdiff = eigv_left_shiftfront_idx_3d
      else:
        xdiff = eigv_diff_0
    elif "shiftright" in btype:
      if "shiftrighttob" in btype:
        xdiff = eigv_right_shiftbottom_idx_3d
      elif "shiftrighttot" in btype:
        xdiff = eigv_right_shifttop_idx_3d
      elif "shiftrighttore" in btype:
        xdiff = eigv_right_shiftrear_idx_3d
      elif "shiftrighttof" in btype:
        xdiff = eigv_right_shiftfront_idx_3d
      else:
        xdiff = eigv_diff_0
    elif "left" in btype:
      xdiff = eigv_left_idx_3d
    elif "right" in btype:
      xdiff = eigv_right_idx_3d
    else:
      xdiff = eigv_gradx_idx_3d
    if "shifty" in btype:
      if "shiftytol" in btype:
        ydiff = eigv_grady_shiftleft_idx_3d
      elif "shiftytori" in btype:
        ydiff = eigv_grady_shiftright_idx_3d
      elif "shiftytore" in btype:
        ydiff = eigv_grady_shiftrear_idx_3d
      elif "shiftytof" in btype:
        ydiff = eigv_grady_shiftfront_idx_3d
      else:
        ydiff = eigv_diff_0
    elif "shiftbottom" in btype:
      if "shiftbottomtol" in btype:
        ydiff = eigv_bottom_shiftleft_idx_3d
      elif "shiftbottomtori" in btype:
        ydiff = eigv_bottom_shiftright_idx_3d
      elif "shiftbottomtore" in btype:
        ydiff = eigv_bottom_shiftrear_idx_3d
      elif "shiftbottomtof" in btype:
        ydiff = eigv_bottom_shiftfront_idx_3d
      else:
        ydiff = eigv_diff_0
    elif "shifttop" in btype:
      if "shifttoptol" in btype:
        ydiff = eigv_top_shiftleft_idx_3d
      elif "shifttoptori" in btype:
        ydiff = eigv_top_shiftright_idx_3d
      elif "shifttoptore" in btype:
        ydiff = eigv_top_shiftrear_idx_3d
      elif "shifttoptof" in btype:
        ydiff = eigv_top_shiftfront_idx_3d
      else:
        ydiff = eigv_diff_0
    elif "bottom" in btype:
      ydiff = eigv_bottom_idx_3d
    elif "top" in btype:
      ydiff = eigv_top_idx_3d
    else:
      ydiff = eigv_grady_idx_3d
    if "shiftz" in btype:
      if "shiftztol" in btype:
        zdiff = eigv_gradz_shiftleft_idx_3d
      elif "shiftztori" in btype:
        zdiff = eigv_gradz_shiftright_idx_3d
      elif "shiftztob" in btype:
        zdiff = eigv_gradz_shiftbottom_idx_3d
      elif "shiftztot" in btype:
        zdiff = eigv_gradz_shifttop_idx_3d
      else:
        zdiff = eigv_diff_0
    elif "shiftrear" in btype:
      if "shiftreartol" in btype:
        zdiff = eigv_rear_shiftleft_idx_3d
      elif "shiftreartori" in btype:
        zdiff = eigv_rear_shiftright_idx_3d
      elif "shiftreartob" in btype:
        zdiff = eigv_rear_shiftbottom_idx_3d
      elif "shiftreartot" in btype:
        zdiff = eigv_rear_shifttop_idx_3d
      else:
        zdiff = eigv_diff_0
    elif "shiftfront" in btype:
      if "shiftfronttol" in btype:
        zdiff = eigv_front_shiftleft_idx_3d
      elif "shiftfronttori" in btype:
        zdiff = eigv_front_shiftright_idx_3d
      elif "shiftfronttob" in btype:
        zdiff = eigv_front_shiftbottom_idx_3d
      elif "shiftfronttot" in btype:
        zdiff = eigv_front_shifttop_idx_3d
      else:
        zdiff = eigv_diff_0
    elif "rear" in btype:
      zdiff = eigv_rear_idx_3d
    elif "front" in btype:
      zdiff = eigv_front_idx_3d
    else:
      zdiff = eigv_gradz_idx_3d
  return(xdiff, ydiff, zdiff)
# end select_eigv_diffs_3d

def diff_0(*args):
  # convenience function that returns 0 always
  return(0)

def eigv_diff_0(*args):
  # convenience function that returns 0,0,0 always
  return(0,0,0)

def fwd_diff_2d(a):
  # return the forward diff in the x and the y direction
  fwd_x = np.zeros_like(a)
  fwd_y = np.zeros_like(a)
  fwd_x[:-1,:] = a[1:,:] - a[:-1,:]
  fwd_y[:,:-1] = a[:,1:] - a[:,:-1]
  return(fwd_x, fwd_y)

def back_diff_2d(a):
  # return the backward diff in the x and the y direction
  back_x = np.zeros_like(a)
  back_y = np.zeros_like(a)
  back_x[1:,:] = a[1:,:] - a[:-1,:]
  back_y[:,1:] = a[:,1:] - a[:,:-1]
  return(back_x, back_y)

def gradient_idx_2d(a, idx):
  # compute gradient ala np.gradient for indices returned by np.where() calls
  # example: gradient_idx_2d(a, np.where(a < 10))
  # WARNING! no bounds checking here, ensure that idx is at least 1 away from each edge of array
  grad_x = np.zeros_like(a)
  grad_y = np.zeros_like(a)
  grad_x[idx[0], idx[1]] = 0.5 * (a[idx[0]+1, idx[1]] - a[idx[0]-1, idx[1]])
  grad_y[idx[0], idx[1]] = 0.5 * (a[idx[0], idx[1]+1] - a[idx[0], idx[1]-1])
  return(grad_x, grad_y)

def gradx_idx_2d(a, idx):
  # compute gradient in x direction ala np.gradient for indices returned by np.where() calls
  # example: gradx_idx_2d(a, np.where(a < 10))
  # WARNING! no bounds checking here, ensure that idx is at least 1 away from each edge of array in x direction
  #grad_x = np.zeros_like(a)
  #grad_x[idx[0], idx[1]] = 0.5 * (a[idx[0]+1, idx[1]] - a[idx[0]-1, idx[1]])
  grad_x = 0.5 * (a[idx[0]+1, idx[1]] - a[idx[0]-1, idx[1]])
  return(grad_x)

def grady_idx_2d(a, idx):
  # compute gradient in y direction ala np.gradient for indices returned by np.where() calls
  # example: grady_idx_2d(a, np.where(a < 10))
  # WARNING! no bounds checking here, ensure that idx is at least 1 away from each edge of array in y direction
  #grad_y = np.zeros_like(a)
  #grad_y[idx[0], idx[1]] = 0.5 * (a[idx[0], idx[1]+1] - a[idx[0], idx[1]-1])
  grad_y = 0.5 * (a[idx[0], idx[1]+1] - a[idx[0], idx[1]-1])
  return(grad_y)

def gradxx_idx_2d(a, idx):
  # compute second derivative in x direction ala np.gradient for indices returned by np.where() calls
  # example: gradx_idx_2d(a, np.where(a < 10))
  # WARNING! no bounds checking here, ensure that idx is at least 1 away from each edge of array in x direction
  grad_xx = (a[idx[0]+1, idx[1]] - 2 * a[idx[0], idx[1]] + a[idx[0]-1, idx[1]])
  return(grad_xx)

def gradyy_idx_2d(a, idx):
  # compute second derivative in y direction ala np.gradient for indices returned by np.where() calls
  # example: grady_idx_2d(a, np.where(a < 10))
  # WARNING! no bounds checking here, ensure that idx is at least 1 away from each edge of array in y direction
  grad_yy = (a[idx[0], idx[1]+1] - 2 * a[idx[0], idx[1]] + a[idx[0], idx[1]-1])
  return(grad_yy)

def left_diff_2d(a):
  # return the derivative in the x direction using only terms to the right
  # called left, because used for case when left direction (i-1) is out of bounds
  left_x = np.zeros_like(a)
  left_x[:-2,:] = -0.5 * a[2:,:] + 2.0 * a[1:-1,:] - 1.5 * a[:-2,:]
  return(left_x)

def left_diff_idx_2d(a, idx):
  # return the derivative in the x direction using only terms to the right
  # called left, because used for case when left direction (i-1) is out of bounds
  # compute for indices returned by np.where() calls
  # example: left_diff_idx_2d(a, np.where(a < 10))
  # WARNING! no bounds checking here, ensure that idx[0] is at least 3 less than array size in x direction
  #left_x = np.zeros_like(a)
  #left_x[idx[0], idx[1]] = -0.5 * a[idx[0]+2, idx[1]] + 2.0 * a[idx[0]+1, idx[1]] - 1.5 * a[idx[0], idx[1]]
  left_x = -0.5 * a[idx[0]+2, idx[1]] + 2.0 * a[idx[0]+1, idx[1]] - 1.5 * a[idx[0], idx[1]]
  return(left_x)

def right_diff_2d(a):
  # return the derivative in the x direction using only terms to the left
  # called right, because used for case when right direction (i+1) is out of bounds
  right_x = np.zeros_like(a)
  right_x[2:,:] = 0.5 * a[:-2,:] - 2.0 * a[1:-1,:] + 1.5 * a[2:,:]
  return(right_x)

def right_diff_idx_2d(a, idx):
  # return the derivative in the x direction using only terms to the left
  # called right, because used for case when right direction (i+1) is out of bounds
  # compute for indices returned by np.where() calls
  # example: right_diff_idx_2d(a, np.where(a < 10))
  # WARNING! no bounds checking here, ensure that idx[0] is at least 2 in the x direction
  #right_x = np.zeros_like(a)
  #right_x[idx[0]+2, idx[1]] = 0.5 * a[idx[0], idx[1]] - 2.0 * a[idx[0]+1, idx[1]] + 1.5 * a[idx[0]+2, idx[1]]
  right_x = 0.5 * a[idx[0]-2, idx[1]] - 2.0 * a[idx[0]-1, idx[1]] + 1.5 * a[idx[0], idx[1]]
  return(right_x)
  
def bottom_diff_2d(a):
  # return the derivative in the y direction using only terms to the top (increasing y direction)
  # called bottom, because used for case when bottom direction (j-1) is out of bounds
  bot_y = np.zeros_like(a)
  bot_y[:,:-2] = -0.5 * a[:,2:] + 2.0 * a[:,1:-1] - 1.5 * a[:,:-2]
  return(bot_y)

def bottom_diff_idx_2d(a, idx):
  # return the derivative in the y direction using only terms to the top (increasing y direction)
  # called bottom, because used for case when bottom direction (j-1) is out of bounds
  # compute for indices returned by np.where() calls
  # example: bottom_diff_idx_2d(a, np.where(a < 10))
  # WARNING! no bounds checking here, ensure that idx[1] is at least 3 less than array size in y direction
  #bot_y = np.zeros_like(a)
  #bot_y[idx[0],idx[1]] = -0.5 * a[idx[0], idx[1]+2] + 2.0 * a[idx[0],idx[1]+1] - 1.5 * a[idx[0],idx[1]]
  bot_y = -0.5 * a[idx[0], idx[1]+2] + 2.0 * a[idx[0],idx[1]+1] - 1.5 * a[idx[0],idx[1]]
  return(bot_y)

def top_diff_2d(a):
  # return the derivative in the y direction using only terms to the bottom (decreasing y direction)
  # called top, because used for case when top direction (j+1) is out of bounds
  top_y = np.zeros_like(a)
  top_y[:,2:] = 0.5 * a[:,:-2] - 2.0 * a[:,1:-1] + 1.5 * a[:,2:]
  return(top_y)

def top_diff_idx_2d(a, idx):
  # return the derivative in the y direction using only terms to the bottom (decreasing y direction)
  # called top, because used for case when top direction (j+1) is out of bounds
  # compute for indices returned by np.where() calls
  # example: top_diff_idx_2d(a, np.where(a < 10))
  # WARNING! no bounds checking here, ensure that idx[1] is at least 2 in y direction
  #top_y = np.zeros_like(a)
  #top_y[idx[0], idx[1]+2] = 0.5 * a[idx[0], idx[1]] - 2.0 * a[idx[0], idx[1]+1] + 1.5 * a[idx[0], idx[1]+2]
  top_y = 0.5 * a[idx[0], idx[1]-2] - 2.0 * a[idx[0], idx[1]-1] + 1.5 * a[idx[0], idx[1]]
  return(top_y)

#################################################
# Find gradient of image inside mask region     #
# Using appropriate left,right,top,bottom diffs #
# at the boundary                               #
#################################################
def gradient_mask_2d(img, mask):
  # Find the gradient of img inside the mask regions
  # Assumes it's a scalar 2D image
  # Returns grad_x, grad_y
  
  # We call this twice because first time modifies the mask
  # TODO see if we can handle the nondifferentiable boundary types with more grace
  # Its really that we need a second order scheme for really small structures.
  bdry_type, bdry_idx, bdry_map = mo.determine_boundary_2d(mask)
  bdry_type, bdry_idx, bdry_map = mo.determine_boundary_2d(mask)

  gradX, gradY = gradient_bdry_2d(img, bdry_idx, bdry_map)
#   return torch.from_numpy(gradX), torch.from_numpy(gradY)
  return gradX, gradY
# def end gradient_mask_2d

def gradient_bdry_2d(img, bdry_idx, bdry_map):

  xsz = img.shape[0]
  ysz = img.shape[1]
  gradX = np.zeros((xsz,ysz))
  gradY = np.zeros((xsz,ysz))

  for btype, bnum in bdry_map.items():
    if bnum == 0:
      # skip, since outside
      continue

    b_idx = bdry_idx[bnum]
    
    if btype[0:8] == "interior":
      gradX[b_idx[0], b_idx[1]] = gradx_idx_2d(img, b_idx)
      gradY[b_idx[0], b_idx[1]] = grady_idx_2d(img, b_idx)
      
    elif btype == "left":
      gradX[b_idx[0], b_idx[1]] = left_diff_idx_2d(img, b_idx)
      gradY[b_idx[0], b_idx[1]] = grady_idx_2d(img, b_idx)

    elif btype == "bottomleft":
      gradX[b_idx[0], b_idx[1]] = left_diff_idx_2d(img, b_idx)
      gradY[b_idx[0], b_idx[1]] = bottom_diff_idx_2d(img, b_idx)

    elif btype == "topleft":
      gradX[b_idx[0], b_idx[1]] = left_diff_idx_2d(img, b_idx)
      gradY[b_idx[0], b_idx[1]] = top_diff_idx_2d(img, b_idx)

    elif btype == "notright":
      gradX[b_idx[0], b_idx[1]] = left_diff_idx_2d(img, b_idx)
      gradY[b_idx[0], b_idx[1]] = grady_idx_2d(img, [b_idx[0]+1, b_idx[1]])

    elif btype == "right":
      gradX[b_idx[0], b_idx[1]] = right_diff_idx_2d(img, b_idx)
      gradY[b_idx[0], b_idx[1]] = grady_idx_2d(img, b_idx)

    elif btype == "bottomright":
      gradX[b_idx[0], b_idx[1]] = right_diff_idx_2d(img, b_idx)
      gradY[b_idx[0], b_idx[1]] = bottom_diff_idx_2d(img, b_idx)

    elif btype == "topright":
      gradX[b_idx[0], b_idx[1]] = right_diff_idx_2d(img, b_idx)
      gradY[b_idx[0], b_idx[1]] = top_diff_idx_2d(img, b_idx)

    elif btype == "notleft":
      gradX[b_idx[0], b_idx[1]] = right_diff_idx_2d(img, b_idx)
      gradY[b_idx[0], b_idx[1]] = grady_idx_2d(img, [b_idx[0]-1, b_idx[1]])

    elif btype == "bottom":
      gradX[b_idx[0], b_idx[1]] = gradx_idx_2d(img, b_idx)
      gradY[b_idx[0], b_idx[1]] = bottom_diff_idx_2d(img, b_idx)

    elif btype == "nottop":
      gradX[b_idx[0], b_idx[1]] = gradx_idx_2d(img, [b_idx[0], b_idx[1]+1])
      gradY[b_idx[0], b_idx[1]] = bottom_diff_idx_2d(img, b_idx)
      
    elif btype == "top":
      gradX[b_idx[0], b_idx[1]] = gradx_idx_2d(img, b_idx)
      gradY[b_idx[0], b_idx[1]] = top_diff_idx_2d(img, b_idx)

    elif btype == "notbottom":
      gradX[b_idx[0], b_idx[1]] = gradx_idx_2d(img, [b_idx[0], b_idx[1]-1])
      gradY[b_idx[0], b_idx[1]] = top_diff_idx_2d(img, b_idx)

    elif btype == "outside":
      # outside mask, skip
      pass

    else:
      # unrecognized type
      print(btype, "unrecognized.  Skipping")
      
  return(gradX, gradY)
# end gradient_mask_2d  


################################
# Begin eigenvector diff opers #
################################
# Following 2D operators are for derivatives of eigenvector fields
# The specialized code here is to handle the natural ambiguities in eigenvector direction (v vs -v)

def eigv_gradx_idx_2d(eigvecs, idx):
  # Return the gradient of each component of the eigenvector taken in the x direction
  gradx_delx = np.zeros(len(idx[0]))
  grady_delx = np.zeros(len(idx[0]))
  
  for ii in range(len(idx[0])):
    xx = idx[0][ii]
    yy = idx[1][ii]
    pix = eigvecs[xx,yy]
    left = eigvecs[xx-1,yy]
    right = eigvecs[xx+1,yy]

    if np.dot(right, left) < 0:
      left = -left
    #if np.dot(pix, right) < 0:
    #  right = -right

    gradx_delx[ii] = 0.5 * (right[0] - left[0])
    grady_delx[ii] = 0.5 * (right[1] - left[1])
  return(gradx_delx, grady_delx)

def eigv_grady_idx_2d(eigvecs, idx):
  # Return the gradient of each component of the eigenvector taken in the y direction
  gradx_dely = np.zeros(len(idx[0]))
  grady_dely = np.zeros(len(idx[0]))
  
  for ii in range(len(idx[0])):
    xx = idx[0][ii]
    yy = idx[1][ii]
    pix = eigvecs[xx,yy]
    bottom = eigvecs[xx,yy-1]
    top = eigvecs[xx,yy+1]

    if np.dot(top, bottom) < 0:
      bottom = -bottom
    #if np.dot(pix, top) < 0:
    #  top = -top

    gradx_dely[ii] = 0.5 * (top[0] - bottom[0])
    grady_dely[ii] = 0.5 * (top[1] - bottom[1])
  return(gradx_dely, grady_dely)

def eigv_left_idx_2d(eigvecs, idx):
  # Return the gradient of each component of the eigenvector taken in the x direction
  # using only terms to the right
  # called left, because used for case when left direction (i-1) is out of bounds
  leftx_delx = np.zeros(len(idx[0]))
  lefty_delx = np.zeros(len(idx[0]))
  
  for ii in range(len(idx[0])):
    xx = idx[0][ii]
    yy = idx[1][ii]
    pix = eigvecs[xx,yy]
    right1 = eigvecs[xx+1,yy]
    right2 = eigvecs[xx+2,yy]

    if np.dot(pix, right1) < 0:
      right1 = -right1
    if np.dot(pix, right2) < 0:
      right2 = -right2
    
    #left_x = -0.5 * a[idx[0]+2, idx[1]] + 2.0 * a[idx[0]+1, idx[1]] - 1.5 * a[idx[0], idx[1]]
    leftx_delx[ii] = -0.5 * right2[0] + 2.0 * right1[0] - 1.5 * pix[0]
    lefty_delx[ii] = -0.5 * right2[1] + 2.0 * right1[1] - 1.5 * pix[1]
  return(leftx_delx, lefty_delx)

def eigv_right_idx_2d(eigvecs, idx):
  # Return the gradient of each component of the eigenvector taken in the x direction
  # using only terms to the left
  # called right, because used for case when right direction (i+1) is out of bounds
  rightx_delx = np.zeros(len(idx[0]))
  righty_delx = np.zeros(len(idx[0]))
  
  for ii in range(len(idx[0])):
    xx = idx[0][ii]
    yy = idx[1][ii]
    pix = eigvecs[xx,yy]
    left1 = eigvecs[xx-1,yy]
    left2 = eigvecs[xx-2,yy]

    if np.dot(pix, left1) < 0:
      left1 = -left1
    if np.dot(pix, left2) < 0:
      left2 = -left2
    
    #right_x = 0.5 * a[idx[0]+2, idx[1]] - 2.0 * a[idx[0]+1, idx[1]] + 1.5 * a[idx[0], idx[1]]
    rightx_delx[ii] = 0.5 * left2[0] - 2.0 * left1[0] + 1.5 * pix[0]
    righty_delx[ii] = 0.5 * left2[1] - 2.0 * left1[1] + 1.5 * pix[1]
  return(rightx_delx, righty_delx)

def eigv_bottom_idx_2d(eigvecs, idx):
  # Return the gradient of each component of the eigenvector taken in the y direction
  # using only terms to the top
  # called bottom, because used for case when bottom direction (j-1) is out of bounds
  bottomx_dely = np.zeros(len(idx[0]))
  bottomy_dely = np.zeros(len(idx[0]))
  
  for ii in range(len(idx[0])):
    xx = idx[0][ii]
    yy = idx[1][ii]
    pix = eigvecs[xx,yy]
    top1 = eigvecs[xx,yy+1]
    top2 = eigvecs[xx,yy+2]

    if np.dot(pix, top1) < 0:
      top1 = -top1
    if np.dot(pix, top2) < 0:
      top2 = -top2
    
    #bottom_y = -0.5 * a[idx[0], idx[1]+2] + 2.0 * a[idx[0], idx[1]+1] - 1.5 * a[idx[0], idx[1]]
    bottomx_dely[ii] = -0.5 * top2[0] + 2.0 * top1[0] - 1.5 * pix[0]
    bottomy_dely[ii] = -0.5 * top2[1] + 2.0 * top1[1] - 1.5 * pix[1]
  return(bottomx_dely, bottomy_dely)

def eigv_top_idx_2d(eigvecs, idx):
  # Return the gradient of each component of the eigenvector taken in the y direction
  # using only terms to the bottom
  # called top, because used for case when top direction (j+1) is out of bounds
  topx_dely = np.zeros(len(idx[0]))
  topy_dely = np.zeros(len(idx[0]))
  
  for ii in range(len(idx[0])):
    xx = idx[0][ii]
    yy = idx[1][ii]
    pix = eigvecs[xx,yy]
    bottom1 = eigvecs[xx,yy-1]
    bottom2 = eigvecs[xx,yy-2]

    if np.dot(pix, bottom1) < 0:
      bottom1 = -bottom1
    if np.dot(pix, bottom2) < 0:
      bottom2 = -bottom2
    
    #top_y = 0.5 * a[idx[0]+2, idx[1]] - 2.0 * a[idx[0]+1, idx[1]] + 1.5 * a[idx[0], idx[1]]
    topx_dely[ii] = 0.5 * bottom2[0] - 2.0 * bottom1[0] + 1.5 * pix[0]
    topy_dely[ii] = 0.5 * bottom2[1] - 2.0 * bottom1[1] + 1.5 * pix[1]
  return(topx_dely, topy_dely)

def eigv_bottomleft_idx_2d(eigvecs, idx):
  # WARNING! bottomleft, topleft, bottomright, topright greatly reduce performance for analytic annulus case!
  # USE AT OWN RISK!!!
  # Return the gradient of each component of the eigenvector taken in the x and y directions
  # using only terms to the right and top
  # (dx0/dx, dx1/dx, dx0/dy, dx1/dy)
  # called bottomleft, because used for case when left and bottom directions (i-1),(j-1) are out of bounds
  leftx_delx = np.zeros(len(idx[0]))
  lefty_delx = np.zeros(len(idx[0]))
  bottomx_dely = np.zeros(len(idx[0]))
  bottomy_dely = np.zeros(len(idx[0]))
  
  for ii in range(len(idx[0])):
    xx = idx[0][ii]
    yy = idx[1][ii]
    pix = eigvecs[xx,yy]
    right = eigvecs[xx+1,yy]
    top = eigvecs[xx,yy+1]
    topright = eigvecs[xx+1,yy+1]

    if np.dot(pix, right) < 0:
      right = -right
    if np.dot(pix, top) < 0:
      top = -top
    if np.dot(pix, topright) < 0:
      topright = -topright
    
    leftx_delx[ii] = 0.5 * (right[0] - pix[0] + topright[0] - top[0])
    lefty_delx[ii] = 0.5 * (right[1] - pix[1] + topright[1] - top[1])
    bottomx_dely[ii] = 0.5 * (top[0] - pix[0] + topright[0] - right[0])
    bottomy_dely[ii] = 0.5 * (top[1] - pix[1] + topright[1] - right[1])
  return(leftx_delx, lefty_delx, bottomx_dely, bottomy_dely)

def eigv_topleft_idx_2d(eigvecs, idx):
  # WARNING! bottomleft, topleft, bottomright, topright greatly reduce performance for analytic annulus case!
  # USE AT OWN RISK!!!
  # Return the gradient of each component of the eigenvector taken in the x and y directions
  # using only terms to the bottom and top
  # (dx0/dx, dx1/dx, dx0/dy, dx1/dy)
  # called topleft, because used for case when left and top directions (i-1),(j+1) are out of bounds
  leftx_delx = np.zeros(len(idx[0]))
  lefty_delx = np.zeros(len(idx[0]))
  topx_dely = np.zeros(len(idx[0]))
  topy_dely = np.zeros(len(idx[0]))
  
  for ii in range(len(idx[0])):
    xx = idx[0][ii]
    yy = idx[1][ii]
    pix = eigvecs[xx,yy]
    right = eigvecs[xx+1,yy]
    bottom = eigvecs[xx,yy-1]
    bottomright = eigvecs[xx+1,yy-1]

    if np.dot(pix, right) < 0:
      right = -right
    if np.dot(pix, bottom) < 0:
      bottom = -bottom
    if np.dot(pix, bottomright) < 0:
      bottomright = -bottomright
    
    leftx_delx[ii] = 0.5 * (right[0] - pix[0] + bottomright[0] - bottom[0])
    lefty_delx[ii] = 0.5 * (right[1] - pix[1] + bottomright[1] - bottom[1])
    topx_dely[ii] = 0.5 * (pix[0] - bottom[0] + right[0] - bottomright[0])
    topy_dely[ii] = 0.5 * (pix[1] - bottom[1] + right[1] - bottomright[1])

  return(leftx_delx, lefty_delx, topx_dely, topy_dely)

def eigv_bottomright_idx_2d(eigvecs, idx):
  # WARNING! bottomleft, topleft, bottomright, topright greatly reduce performance for analytic annulus case!
  # USE AT OWN RISK!!!
  # Return the gradient of each component of the eigenvector taken in the y direction
  # using only terms to the top
  # called bottom, because used for case when bottom direction (j-1) is out of bounds
  rightx_delx = np.zeros(len(idx[0]))
  righty_delx = np.zeros(len(idx[0]))
  bottomx_dely = np.zeros(len(idx[0]))
  bottomy_dely = np.zeros(len(idx[0]))
  
  for ii in range(len(idx[0])):
    xx = idx[0][ii]
    yy = idx[1][ii]
    pix = eigvecs[xx,yy]
    left = eigvecs[xx-1,yy]
    top = eigvecs[xx,yy+1]
    topleft = eigvecs[xx-1,yy+1]

    if np.dot(pix, left) < 0:
      left = -left
    if np.dot(pix, top) < 0:
      top = -top
    if np.dot(pix, topleft) < 0:
      topleft = -topleft
    
    rightx_delx[ii] = 0.5 * (pix[0] - left[0] + top[0] - topleft[0])
    righty_delx[ii] = 0.5 * (pix[1] - left[1] + top[1] - topleft[1])
    bottomx_dely[ii] = 0.5 * (top[0] - pix[0] + topleft[0] - left[0])
    bottomy_dely[ii] = 0.5 * (top[1] - pix[1] + topleft[1] - left[1])

  return(rightx_delx, righty_delx, bottomx_dely, bottomy_dely)

def eigv_topright_idx_2d(eigvecs, idx):
  # WARNING! bottomleft, topleft, bottomright, topright greatly reduce performance for analytic annulus case!
  # USE AT OWN RISK!!!
  # Return the gradient of each component of the eigenvector taken in the x direction
  # using only terms to the left
  # called right, because used for case when right direction (i+1) is out of bounds
  rightx_delx = np.zeros(len(idx[0]))
  righty_delx = np.zeros(len(idx[0]))
  topx_dely = np.zeros(len(idx[0]))
  topy_dely = np.zeros(len(idx[0]))
  
  for ii in range(len(idx[0])):
    xx = idx[0][ii]
    yy = idx[1][ii]
    pix = eigvecs[xx,yy]
    left = eigvecs[xx-1,yy]
    bottom = eigvecs[xx,yy-1]
    bottomleft = eigvecs[xx-1,yy-1]

    if np.dot(pix, left) < 0:
      left = -left
    if np.dot(pix, bottom) < 0:
      bottom = -bottom
    if np.dot(pix, bottomleft) < 0:
      bottomleft = -bottomleft
    
    rightx_delx[ii] = 0.5 * (pix[0] - left[0] + bottom[0] - bottomleft[0])
    righty_delx[ii] = 0.5 * (pix[1] - left[1] + bottom[1] - bottomleft[1])
    topx_dely[ii] = 0.5 * (pix[0] - bottom[0] + left[0] - bottomleft[0])
    topy_dely[ii] = 0.5 * (pix[1] - bottom[1] + left[1] - bottomleft[1])

  return(rightx_delx, righty_delx, topx_dely, topy_dely)

#######################
# Begin 3D diff opers #
#######################
def fwd_diff_3d(a):
  # return the forward diff in the x, y and z direction
  fwd_x = np.zeros_like(a)
  fwd_y = np.zeros_like(a)
  fwd_z = np.zeros_like(a)
  fwd_x[:-1,:,:] = a[1:,:,:] - a[:-1,:,:]
  fwd_y[:,:-1,:] = a[:,1:,:] - a[:,:-1,:]
  fwd_z[:,:,:-1] = a[:,:,1:] - a[:,:,:-1]
  return(fwd_x, fwd_y, fwd_z)

def back_diff_3d(a):
  # return the backward diff in the x, y and z direction
  back_x = np.zeros_like(a)
  back_y = np.zeros_like(a)
  back_z = np.zeros_like(a)
  back_x[1:,:,:] = a[1:,:,:] - a[:-1,:,:]
  back_y[:,1:,:] = a[:,1:,:] - a[:,:-1,:]
  back_z[:,:,1:] = a[:,:,1:] - a[:,:,:-1]
  return(back_x, back_y, back_z)

def gradient_idx_3d(a, idx):
  # compute gradient ala np.gradient for indices returned by np.where() calls
  # example: gradient_idx_2d(a, np.where(a < 10))
  # WARNING! no bounds checking here, ensure that idx is at least 1 away from each edge of array
  grad_x = np.zeros_like(a)
  grad_y = np.zeros_like(a)
  grad_z = np.zeros_like(a)
  grad_x[idx[0], idx[1], idx[2]] = 0.5 * (a[idx[0]+1, idx[1], idx[2]] - a[idx[0]-1, idx[1], idx[2]])
  grad_y[idx[0], idx[1], idx[2]] = 0.5 * (a[idx[0], idx[1]+1, idx[2]] - a[idx[0], idx[1]-1, idx[2]])
  grad_z[idx[0], idx[1], idx[2]] = 0.5 * (a[idx[0], idx[1], idx[2]+1] - a[idx[0], idx[1], idx[2]-1])
  return(grad_x, grad_y, grad_z)

def gradx_idx_3d(a, idx):
  # compute gradient in x direction ala np.gradient for indices returned by np.where() calls
  # example: gradx_idx_3d(a, np.where(a < 10))
  # WARNING! no bounds checking here, ensure that idx is at least 1 away from each edge of array in x direction
  grad_x = 0.5 * (a[idx[0]+1, idx[1], idx[2]] - a[idx[0]-1, idx[1], idx[2]])
  return(grad_x)

def grady_idx_3d(a, idx):
  # compute gradient in y direction ala np.gradient for indices returned by np.where() calls
  # example: grady_idx_3d(a, np.where(a < 10))
  # WARNING! no bounds checking here, ensure that idx is at least 1 away from each edge of array in y direction
  grad_y = 0.5 * (a[idx[0], idx[1]+1, idx[2]] - a[idx[0], idx[1]-1, idx[2]])
  return(grad_y)

def gradz_idx_3d(a, idx):
  # compute gradient in z direction ala np.gradient for indices returned by np.where() calls
  # example: gradz_idx_3d(a, np.where(a < 10))
  # WARNING! no bounds checking here, ensure that idx is at least 1 away from each edge of array in z direction
  grad_z = 0.5 * (a[idx[0], idx[1], idx[2]+1] - a[idx[0], idx[1], idx[2]-1])
  return(grad_z)

def gradxx_idx_3d(a, idx):
  # compute second derivative in x direction ala np.gradient for indices returned by np.where() calls
  # example: gradx_idx_2d(a, np.where(a < 10))
  # WARNING! no bounds checking here, ensure that idx is at least 1 away from each edge of array in x direction
  grad_xx = (a[idx[0]+1, idx[1], idx[2]] - 2 * a[idx[0], idx[1], idx[2]] + a[idx[0]-1, idx[1], idx[2]])
  return(grad_xx)

def gradyy_idx_3d(a, idx):
  # compute second derivative in y direction ala np.gradient for indices returned by np.where() calls
  # example: grady_idx_2d(a, np.where(a < 10))
  # WARNING! no bounds checking here, ensure that idx is at least 1 away from each edge of array in y direction
  grad_yy = (a[idx[0], idx[1]+1, idx[2]] - 2 * a[idx[0], idx[1], idx[2]] + a[idx[0], idx[1]-1, idx[2]])
  return(grad_yy)

def gradzz_idx_3d(a, idx):
  # compute second derivative in z direction ala np.gradient for indices returned by np.where() calls
  # example: grady_idx_2d(a, np.where(a < 10))
  # WARNING! no bounds checking here, ensure that idx is at least 1 away from each edge of array in y direction
  grad_zz = (a[idx[0], idx[1], idx[2]+1] - 2 * a[idx[0], idx[1], idx[2]] + a[idx[0], idx[1], idx[2]-1])
  return(grad_zz)

def gradxy_bottomrightdiag_idx_3d(a, idx):
  # Compute cross second derivative gradxgrady
  # WARNING! no bounds checking here, ensure that idx is at least 1 away from each edge of array in y direction
  grad_xy = (-0.5 * a[idx[0]-1, idx[1]+1, idx[2]] - 0.5 * a[idx[0]+1, idx[1]-1, idx[2]] + 0.5*a[idx[0], idx[1]-1, idx[2]]
            + 0.5 * a[idx[0]+1, idx[1], idx[2]] + 0.5 * a[idx[0]-1, idx[1], idx[2]] + 0.5*a[idx[0], idx[1]+1, idx[2]]
            - a[idx[0], idx[1], idx[2]] )
  return(grad_xy)

def gradxy_bottomleftdiag_idx_3d(a, idx):
  # Compute cross second derivative gradxgrady
  # WARNING! no bounds checking here, ensure that idx is at least 1 away from each edge of array in y direction
  grad_xy = (0.5 * a[idx[0]+1, idx[1]+1, idx[2]] + 0.5 * a[idx[0]-1, idx[1]-1, idx[2]] - 0.5*a[idx[0], idx[1]-1, idx[2]]
            - 0.5 * a[idx[0]+1, idx[1], idx[2]] - 0.5 * a[idx[0]-1, idx[1], idx[2]] - 0.5*a[idx[0], idx[1]+1, idx[2]]
            + a[idx[0], idx[1], idx[2]] )
  return(grad_xy)

def gradxy_bottomleft_idx_3d(a, idx):
  # Compute cross second derivative gradxgrady
  # WARNING! no bounds checking here, ensure that idx is at least 1 away from each edge of array in y direction
  grad_xy = (a[idx[0]+1, idx[1]+1, idx[2]] - a[idx[0]+1, idx[1], idx[2]]
            - a[idx[0], idx[1]+1, idx[2]] + a[idx[0], idx[1], idx[2]])
  return(grad_xy)

def gradxy_bottomright_idx_3d(a, idx):
  # Compute cross second derivative gradxgrady
  # WARNING! no bounds checking here, ensure that idx is at least 1 away from each edge of array in y direction
  grad_xy = (-a[idx[0]-1, idx[1]+1, idx[2]] + a[idx[0]-1, idx[1], idx[2]]
            + a[idx[0], idx[1]+1, idx[2]] - a[idx[0], idx[1], idx[2]])
  return(grad_xy)

def gradxy_topleft_idx_3d(a, idx):
  # Compute cross second derivative gradxgrady
  # WARNING! no bounds checking here, ensure that idx is at least 1 away from each edge of array in y direction
  grad_xy = (-a[idx[0]+1, idx[1]-1, idx[2]] + a[idx[0]+1, idx[1], idx[2]]
            + a[idx[0], idx[1]-1, idx[2]] - a[idx[0], idx[1], idx[2]])
  return(grad_xy)

def gradxy_topright_idx_3d(a, idx):
  # Compute cross second derivative gradxgrady
  # WARNING! no bounds checking here, ensure that idx is at least 1 away from each edge of array in y direction
  grad_xy = (a[idx[0]-1, idx[1]-1, idx[2]] - a[idx[0]-1, idx[1], idx[2]]
            - a[idx[0], idx[1]-1, idx[2]] + a[idx[0], idx[1], idx[2]])
  return(grad_xy)

def gradxy_left_idx_3d(a, idx):
  # Compute cross second derivative gradxgrady
  # WARNING! no bounds checking here, ensure that idx is at least 1 away from each edge of array in y direction
  grad_xy = (0.5 * a[idx[0]+1, idx[1]+1, idx[2]] - 0.5 * a[idx[0]+1, idx[1]-1, idx[2]]
            + 0.5*a[idx[0], idx[1]-1, idx[2]] - 0.5*a[idx[0], idx[1]+1, idx[2]])
  return(grad_xy)

def gradxy_bottom_idx_3d(a, idx):
  # Compute cross second derivative gradxgrady
  # WARNING! no bounds checking here, ensure that idx is at least 1 away from each edge of array in y direction
  grad_xy = (0.5 * a[idx[0]+1, idx[1]+1, idx[2]] - 0.5 * a[idx[0]-1, idx[1]+1, idx[2]]
            + 0.5*a[idx[0]-1, idx[1], idx[2]] - 0.5*a[idx[0]+1, idx[1], idx[2]])
  return(grad_xy)

def gradxy_right_idx_3d(a, idx):
  # Compute cross second derivative gradxgrady
  # WARNING! no bounds checking here, ensure that idx is at least 1 away from each edge of array in y direction
  grad_xy = (0.5 * a[idx[0]-1, idx[1]-1, idx[2]] - 0.5 * a[idx[0]-1, idx[1]+1, idx[2]]
            - 0.5*a[idx[0], idx[1]-1, idx[2]] + 0.5*a[idx[0], idx[1]+1, idx[2]])
  return(grad_xy)

def gradxy_top_idx_3d(a, idx):
  # Compute cross second derivative gradxgrady
  # WARNING! no bounds checking here, ensure that idx is at least 1 away from each edge of array in y direction
  grad_xy = (0.5 * a[idx[0]-1, idx[1]-1, idx[2]] - 0.5 * a[idx[0]+1, idx[1]-1, idx[2]]
            - 0.5*a[idx[0]-1, idx[1], idx[2]] + 0.5*a[idx[0]+1, idx[1], idx[2]])
  return(grad_xy)

def gradxz_rearrightdiag_idx_3d(a, idx):
  # Compute cross second derivative gradxgradz
  # WARNING! no bounds checking here, ensure that idx is at least 1 away from each edge of array in y direction
  grad_xz = (-0.5 * a[idx[0]-1, idx[1], idx[2]+1] - 0.5 * a[idx[0]+1, idx[1], idx[2]-1] + 0.5*a[idx[0], idx[1], idx[2]-1]
            + 0.5 * a[idx[0]+1, idx[1], idx[2]] + 0.5 * a[idx[0]-1, idx[1], idx[2]] + 0.5*a[idx[0], idx[1], idx[2]+1]
            - a[idx[0], idx[1], idx[2]] )
  return(grad_xz)

def gradxz_rearleftdiag_idx_3d(a, idx):
  # Compute cross second derivative gradxgradz
  # WARNING! no bounds checking here, ensure that idx is at least 1 away from each edge of array in y direction
  grad_xz = (0.5 * a[idx[0]+1, idx[1], idx[2]+1] + 0.5 * a[idx[0]-1, idx[1], idx[2]-1] - 0.5*a[idx[0], idx[1], idx[2]-1]
            - 0.5 * a[idx[0]+1, idx[1], idx[2]] - 0.5 * a[idx[0]-1, idx[1], idx[2]] - 0.5*a[idx[0], idx[1], idx[2]+1]
            + a[idx[0], idx[1], idx[2]] )
  return(grad_xz)

def gradxz_rearleft_idx_3d(a, idx):
  # Compute cross second derivative gradxgradz
  # WARNING! no bounds checking here, ensure that idx is at least 1 away from each edge of array in y direction
  grad_xz = (a[idx[0]+1, idx[1], idx[2]+1] - a[idx[0]+1, idx[1], idx[2]]
            - a[idx[0], idx[1], idx[2]+1] + a[idx[0], idx[1], idx[2]])
  return(grad_xz)

def gradxz_rearright_idx_3d(a, idx):
  # Compute cross second derivative gradxgradz
  # WARNING! no bounds checking here, ensure that idx is at least 1 away from each edge of array in y direction
  grad_xz = (-a[idx[0]-1, idx[1], idx[2]+1] + a[idx[0]-1, idx[1], idx[2]]
            + a[idx[0], idx[1], idx[2]+1] - a[idx[0], idx[1], idx[2]])
  return(grad_xz)

def gradxz_frontleft_idx_3d(a, idx):
  # Compute cross second derivative gradxgradz
  # WARNING! no bounds checking here, ensure that idx is at least 1 away from each edge of array in y direction
  grad_xz = (-a[idx[0]+1, idx[1], idx[2]-1] + a[idx[0]+1, idx[1], idx[2]]
            + a[idx[0], idx[1], idx[2]-1] - a[idx[0], idx[1], idx[2]])
  return(grad_xz)

def gradxz_frontright_idx_3d(a, idx):
  # Compute cross second derivative gradxgradz
  # WARNING! no bounds checking here, ensure that idx is at least 1 away from each edge of array in y direction
  grad_xz = (a[idx[0]-1, idx[1], idx[2]-1] - a[idx[0]-1, idx[1], idx[2]]
            - a[idx[0], idx[1], idx[2]-1] + a[idx[0], idx[1], idx[2]])
  return(grad_xz)

def gradxz_left_idx_3d(a, idx):
  # Compute cross second derivative gradxgradz
  # WARNING! no bounds checking here, ensure that idx is at least 1 away from each edge of array in y direction
  grad_xz = (0.5 * a[idx[0]+1, idx[1], idx[2]+1] - 0.5 * a[idx[0]+1, idx[1], idx[2]-1]
            + 0.5*a[idx[0], idx[1], idx[2]-1] - 0.5*a[idx[0], idx[1], idx[2]+1])
  return(grad_xz)

def gradxz_rear_idx_3d(a, idx):
  # Compute cross second derivative gradxgradz
  # WARNING! no bounds checking here, ensure that idx is at least 1 away from each edge of array in y direction
  grad_xz = (0.5 * a[idx[0]+1, idx[1], idx[2]+1] - 0.5 * a[idx[0]-1, idx[1], idx[2]+1]
            + 0.5*a[idx[0]-1, idx[1], idx[2]] - 0.5*a[idx[0]+1, idx[1], idx[2]])
  return(grad_xz)

def gradxz_right_idx_3d(a, idx):
  # Compute cross second derivative gradxgradz
  # WARNING! no bounds checking here, ensure that idx is at least 1 away from each edge of array in y direction
  grad_xz = (0.5 * a[idx[0]-1, idx[1], idx[2]-1] - 0.5 * a[idx[0]-1, idx[1], idx[2]+1]
            - 0.5*a[idx[0], idx[1], idx[2]-1] + 0.5*a[idx[0], idx[1], idx[2]+1])
  return(grad_xz)

def gradxz_front_idx_3d(a, idx):
  # Compute cross second derivative gradxgradz
  # WARNING! no bounds checking here, ensure that idx is at least 1 away from each edge of array in y direction
  grad_xz = (0.5 * a[idx[0]-1, idx[1], idx[2]-1] - 0.5 * a[idx[0]+1, idx[1], idx[2]-1]
            - 0.5*a[idx[0]-1, idx[1], idx[2]] + 0.5*a[idx[0]+1, idx[1], idx[2]])
  return(grad_xz)

def gradyz_reartopdiag_idx_3d(a, idx):
  # Compute cross second derivative gradygradz
  # WARNING! no bounds checking here, ensure that idx is at least 1 away from each edge of array in y direction
  grad_yz = (-0.5 * a[idx[0], idx[1]-1, idx[2]+1] - 0.5 * a[idx[0], idx[1]+1, idx[2]-1] + 0.5*a[idx[0], idx[1], idx[2]-1]
            + 0.5 * a[idx[0], idx[1]+1, idx[2]] + 0.5 * a[idx[0], idx[1]-1, idx[2]] + 0.5*a[idx[0], idx[1], idx[2]+1]
            - a[idx[0], idx[1], idx[2]] )
  return(grad_yz)

def gradyz_rearbottomdiag_idx_3d(a, idx):
  # Compute cross second derivative gradygradz
  # WARNING! no bounds checking here, ensure that idx is at least 1 away from each edge of array in y direction
  grad_yz = (0.5 * a[idx[0], idx[1]+1, idx[2]+1] + 0.5 * a[idx[0], idx[1]-1, idx[2]-1] - 0.5*a[idx[0], idx[1], idx[2]-1]
            - 0.5 * a[idx[0], idx[1]+1, idx[2]] - 0.5 * a[idx[0], idx[1]-1, idx[2]] - 0.5*a[idx[0], idx[1], idx[2]+1]
            + a[idx[0], idx[1], idx[2]] )
  return(grad_yz)

def gradyz_rearbottom_idx_3d(a, idx):
  # Compute cross second derivative gradygradz
  # WARNING! no bounds checking here, ensure that idx is at least 1 away from each edge of array in y direction
  grad_yz = (a[idx[0], idx[1]+1, idx[2]+1] - a[idx[0], idx[1]+1, idx[2]]
            - a[idx[0], idx[1], idx[2]+1] + a[idx[0], idx[1], idx[2]])
  return(grad_yz)

def gradyz_reartop_idx_3d(a, idx):
  # Compute cross second derivative gradygradz
  # WARNING! no bounds checking here, ensure that idx is at least 1 away from each edge of array in y direction
  grad_yz = (-a[idx[0], idx[1]-1, idx[2]+1] + a[idx[0], idx[1]-1, idx[2]]
            + a[idx[0], idx[1], idx[2]+1] - a[idx[0], idx[1], idx[2]])
  return(grad_yz)

def gradyz_frontbottom_idx_3d(a, idx):
  # Compute cross second derivative gradygradz
  # WARNING! no bounds checking here, ensure that idx is at least 1 away from each edge of array in y direction
  grad_yz = (-a[idx[0], idx[1]+1, idx[2]-1] + a[idx[0], idx[1]+1, idx[2]]
            + a[idx[0], idx[1], idx[2]-1] - a[idx[0], idx[1], idx[2]])
  return(grad_yz)

def gradyz_fronttop_idx_3d(a, idx):
  # Compute cross second derivative gradygradz
  # WARNING! no bounds checking here, ensure that idx is at least 1 away from each edge of array in y direction
  grad_yz = (a[idx[0], idx[1]-1, idx[2]-1] - a[idx[0], idx[1]-1, idx[2]]
            - a[idx[0], idx[1], idx[2]-1] + a[idx[0], idx[1], idx[2]])
  return(grad_yz)

def gradyz_bottom_idx_3d(a, idx):
  # Compute cross second derivative gradygradz
  # WARNING! no bounds checking here, ensure that idx is at least 1 away from each edge of array in y direction
  grad_yz = (0.5 * a[idx[0], idx[1]+1, idx[2]+1] - 0.5 * a[idx[0], idx[1]+1, idx[2]-1]
            + 0.5*a[idx[0], idx[1], idx[2]-1] - 0.5*a[idx[0], idx[1], idx[2]+1])
  return(grad_yz)

def gradyz_rear_idx_3d(a, idx):
  # Compute cross second derivative gradygradz
  # WARNING! no bounds checking here, ensure that idx is at least 1 away from each edge of array in y direction
  grad_yz = (0.5 * a[idx[0], idx[1]+1, idx[2]+1] - 0.5 * a[idx[0], idx[1]-1, idx[2]+1]
            + 0.5*a[idx[0], idx[1]-1, idx[2]] - 0.5*a[idx[0], idx[1]+1, idx[2]])
  return(grad_yz)

def gradyz_top_idx_3d(a, idx):
  # Compute cross second derivative gradygradz
  # WARNING! no bounds checking here, ensure that idx is at least 1 away from each edge of array in y direction
  grad_yz = (0.5 * a[idx[0], idx[1]-1, idx[2]-1] - 0.5 * a[idx[0], idx[1]-1, idx[2]+1]
            - 0.5*a[idx[0], idx[1], idx[2]-1] + 0.5*a[idx[0], idx[1], idx[2]+1])
  return(grad_yz)

def gradyz_front_idx_3d(a, idx):
  # Compute cross second derivative gradygradz
  # WARNING! no bounds checking here, ensure that idx is at least 1 away from each edge of array in y direction
  grad_yz = (0.5 * a[idx[0], idx[1]-1, idx[2]-1] - 0.5 * a[idx[0], idx[1]+1, idx[2]-1]
            - 0.5*a[idx[0], idx[1]-1, idx[2]] + 0.5*a[idx[0], idx[1]+1, idx[2]])
  return(grad_yz)

def gradx_shiftbottom_idx_3d(a, idx):
  # compute gradient in x direction using a shifted y coord ala np.gradient for indices returned by np.where() calls
  # example: gradx_idx_3d(a, np.where(a < 10))
  # 2nd order accurate gradx at [i,j,k] = 0.5a[i+1,j+1,k]-0.5a[i-1,j+1,k]+0.25a[i+1,j+2,k]-0.25a[i-1,j+2,k]
  # WARNING! no bounds checking here, ensure that idx is at least 1 away from each edge of array in x direction
  grad_x = 0.5 * a[idx[0]+1, idx[1]+1, idx[2]] - 0.5 * a[idx[0]-1, idx[1]+1, idx[2]] \
         + 0.25 * a[idx[0]+1, idx[1]+2, idx[2]] - 0.25 * a[idx[0]-1, idx[1]+2, idx[2]]
  return(grad_x)

def gradx_shifttop_idx_3d(a, idx):
  # compute gradient in x direction using a shifted y coord ala np.gradient for indices returned by np.where() calls
  # example: gradx_idx_3d(a, np.where(a < 10))
  # WARNING! no bounds checking here, ensure that idx is at least 1 away from each edge of array in x direction
  # 2nd order accurate gradx at [i,j,k] = a[i+1,j-1,k]-a[i-1,j-1,k]-0.5a[i+1,j-2,k]+0.5a[i-1,j-2,k]
  grad_x = a[idx[0]+1, idx[1]-1, idx[2]] - a[idx[0]-1, idx[1]-1, idx[2]] \
         - 0.5 * a[idx[0]+1, idx[1]-2, idx[2]] + 0.5 * a[idx[0]-1, idx[1]-2, idx[2]]
  return(grad_x)

def gradx_shiftrear_idx_3d(a, idx):
  # compute gradient in x direction using a shifted z coord ala np.gradient for indices returned by np.where() calls
  # example: gradx_idx_3d(a, np.where(a < 10))
  # WARNING! no bounds checking here, ensure that idx is at least 1 away from each edge of array in x direction
  grad_x = 0.5 * a[idx[0]+1, idx[1], idx[2]+1] - 0.5 * a[idx[0]-1, idx[1], idx[2]+1] \
         + 0.25 * a[idx[0]+1, idx[1], idx[2]+2] - 0.25 * a[idx[0]-1, idx[1], idx[2]+2]
  return(grad_x)

def gradx_shiftfront_idx_3d(a, idx):
  # compute gradient in x direction using a shifted z coord ala np.gradient for indices returned by np.where() calls
  # example: gradx_idx_3d(a, np.where(a < 10))
  # WARNING! no bounds checking here, ensure that idx is at least 1 away from each edge of array in x direction
  grad_x = a[idx[0]+1, idx[1], idx[2]-1] - a[idx[0]-1, idx[1], idx[2]-1] \
         - 0.5 * a[idx[0]+1, idx[1], idx[2]-2] + 0.5 * a[idx[0]-1, idx[1], idx[2]-2]
  return(grad_x)

def left_shiftbottom_idx_3d(a, idx):
  # compute gradient in x direction using a shifted y coord ala np.gradient for indices returned by np.where() calls
  # example: gradx_idx_3d(a, np.where(a < 10))
  # 2nd order accurate gradx at [i,j,k] = 3a[i+1,j+1,k] - 0.5a[i+2,j+1,k] - a[i+1,j+2,k]
  #                                       +a[i,j+2,k] - 2.5a[i,j+1,k]
  # WARNING! no bounds checking here, ensure that idx is at least 1 away from each edge of array in x direction
  grad_x = 3 * a[idx[0]+1, idx[1]+1, idx[2]] - 0.5 * a[idx[0]+2, idx[1]+1, idx[2]] - a[idx[0]+1, idx[1]+2, idx[2]] \
         + a[idx[0], idx[1]+2, idx[2]] - 2.5 * a[idx[0], idx[1]+1, idx[2]]
  return(grad_x)

def left_shifttop_idx_3d(a, idx):
  # compute gradient in x direction using a shifted y coord ala np.gradient for indices returned by np.where() calls
  # example: gradx_idx_3d(a, np.where(a < 10))
  # WARNING! no bounds checking here, ensure that idx is at least 1 away from each edge of array in x direction
  grad_x = 3 * a[idx[0]+1, idx[1]-1, idx[2]] - 0.5 * a[idx[0]+2, idx[1]-1, idx[2]] - a[idx[0]+1, idx[1]-2, idx[2]] \
         + a[idx[0], idx[1]-2, idx[2]] - 2.5 * a[idx[0], idx[1]-1, idx[2]]
  return(grad_x)

def left_shiftrear_idx_3d(a, idx):
  # compute gradient in x direction using a shifted z coord ala np.gradient for indices returned by np.where() calls
  # example: gradx_idx_3d(a, np.where(a < 10))
  # WARNING! no bounds checking here, ensure that idx is at least 1 away from each edge of array in x direction
  grad_x = 3 * a[idx[0]+1, idx[1], idx[2]+1] - 0.5 * a[idx[0]+2, idx[1], idx[2]+1] - a[idx[0]+1, idx[1], idx[2]+2] \
         + a[idx[0], idx[1], idx[2]+2] - 2.5 * a[idx[0], idx[1], idx[2]+1]
  return(grad_x)

def left_shiftfront_idx_3d(a, idx):
  # compute gradient in x direction using a shifted z coord ala np.gradient for indices returned by np.where() calls
  # example: gradx_idx_3d(a, np.where(a < 10))
  # WARNING! no bounds checking here, ensure that idx is at least 1 away from each edge of array in x direction
  grad_x = 3 * a[idx[0]+1, idx[1], idx[2]-1] - 0.5 * a[idx[0]+2, idx[1], idx[2]-1] - a[idx[0]+1, idx[1], idx[2]-2] \
         + a[idx[0], idx[1], idx[2]-2] - 2.5 * a[idx[0], idx[1], idx[2]-1]
  return(grad_x)

def right_shiftbottom_idx_3d(a, idx):
  # compute gradient in x direction using a shifted y coord ala np.gradient for indices returned by np.where() calls
  # example: gradx_idx_3d(a, np.where(a < 10))
  # WARNING! no bounds checking here, ensure that idx is at least 1 away from each edge of array in x direction
  # 2nd order accurate gradx at [i,j,k] = -3a[i-1,j+1,k] + 0.5a[i-2,j+1,k] + a[i-1,j+2,k]
  #                                       -a[i,j+2,k] + 2.5a[i,j+1,k]
  # WARNING! no bounds checking here, ensure that idx is at least 1 away from each edge of array in x direction
  grad_x = -3 * a[idx[0]-1, idx[1]+1, idx[2]] + 0.5 * a[idx[0]-2, idx[1]+1, idx[2]] + a[idx[0]-1, idx[1]+2, idx[2]] \
         - a[idx[0], idx[1]+2, idx[2]] + 2.5 * a[idx[0], idx[1]+1, idx[2]]
  return(grad_x)

def right_shifttop_idx_3d(a, idx):
  # compute gradient in x direction using a shifted y coord ala np.gradient for indices returned by np.where() calls
  # example: gradx_idx_3d(a, np.where(a < 10))
  # WARNING! no bounds checking here, ensure that idx is at least 1 away from each edge of array in x direction
  grad_x = -3 * a[idx[0]-1, idx[1]-1, idx[2]] + 0.5 * a[idx[0]-2, idx[1]-1, idx[2]] + a[idx[0]-1, idx[1]-2, idx[2]] \
         - a[idx[0], idx[1]-2, idx[2]] + 2.5 * a[idx[0], idx[1]-1, idx[2]]
  return(grad_x)

def right_shiftrear_idx_3d(a, idx):
  # compute gradient in x direction using a shifted z coord ala np.gradient for indices returned by np.where() calls
  # example: gradx_idx_3d(a, np.where(a < 10))
  # WARNING! no bounds checking here, ensure that idx is at least 1 away from each edge of array in x direction
  grad_x = -3 * a[idx[0]-1, idx[1], idx[2]+1] + 0.5 * a[idx[0]-2, idx[1], idx[2]+1] + a[idx[0]-1, idx[1], idx[2]+2] \
         - a[idx[0], idx[1], idx[2]+2] + 2.5 * a[idx[0], idx[1], idx[2]+1]
  return(grad_x)

def right_shiftfront_idx_3d(a, idx):
  # compute gradient in x direction using a shifted z coord ala np.gradient for indices returned by np.where() calls
  # example: gradx_idx_3d(a, np.where(a < 10))
  # WARNING! no bounds checking here, ensure that idx is at least 1 away from each edge of array in x direction
  grad_x = -3 * a[idx[0]-1, idx[1], idx[2]-1] + 0.5 * a[idx[0]-2, idx[1], idx[2]-1] + a[idx[0]-1, idx[1], idx[2]-2] \
         - a[idx[0], idx[1], idx[2]-2] + 2.5 * a[idx[0], idx[1], idx[2]-1]
  return(grad_x)

def grady_shiftleft_idx_3d(a, idx):
  # compute gradient in y direction using a shifted x coord ala np.gradient for indices returned by np.where() calls
  # example: grady_idx_3d(a, np.where(a < 10))
  # 2nd order accurate grady at [i,j,k] = 0.5a[i+1,j+1,k]-0.5a[i+1,j-1,k]+0.25a[i+2,j+1,k]-0.25a[i+2,j-1,k]
  # WARNING! no bounds checking here, ensure that idx is at least 1 away from each edge of array in x direction
  grad_y = 0.5 * a[idx[0]+1, idx[1]+1, idx[2]] - 0.5 * a[idx[0]+1, idx[1]-1, idx[2]] \
         + 0.25 * a[idx[0]+2, idx[1]+1, idx[2]] - 0.25 * a[idx[0]+2, idx[1]-1, idx[2]]
  return(grad_y)

def grady_shiftright_idx_3d(a, idx):
  # compute gradient in y direction using a shifted x coord ala np.gradient for indices returned by np.where() calls
  # example: grady_idx_3d(a, np.where(a < 10))
  # WARNING! no bounds checking here, ensure that idx is at least 1 away from each edge of array in y direction
  grad_y = a[idx[0]-1, idx[1]+1, idx[2]] - a[idx[0]-1, idx[1]-1, idx[2]] \
         - 0.5 * a[idx[0]-2, idx[1]+1, idx[2]] + 0.5 * a[idx[0]-2, idx[1]-1, idx[2]]
  return(grad_y)

def grady_shiftrear_idx_3d(a, idx):
  # compute gradient in y direction using a shifted z coord ala np.gradient for indices returned by np.where() calls
  # example: grady_idx_3d(a, np.where(a < 10))
  # WARNING! no bounds checking here, ensure that idx is at least 1 away from each edge of array in y direction
  grad_y = 0.5 * a[idx[0], idx[1]+1, idx[2]+1] - 0.5 * a[idx[0], idx[1]-1, idx[2]+1] \
         + 0.25 * a[idx[0], idx[1]+1, idx[2]+2] - 0.25 * a[idx[0], idx[1]-1, idx[2]+2]
  return(grad_y)

def grady_shiftfront_idx_3d(a, idx):
  # compute gradient in y direction using a shifted z coord ala np.gradient for indices returned by np.where() calls
  # example: grady_idx_3d(a, np.where(a < 10))
  # WARNING! no bounds checking here, ensure that idx is at least 1 away from each edge of array in y direction
  grad_y = a[idx[0], idx[1]+1, idx[2]-1] - a[idx[0], idx[1]-1, idx[2]-1] \
         - 0.5 * a[idx[0], idx[1]+1, idx[2]-2] + 0.5 * a[idx[0], idx[1]-1, idx[2]-2]
  return(grad_y)

def bottom_shiftleft_idx_3d(a, idx):
  # compute gradient in y direction using a shifted x coord ala np.gradient for indices returned by np.where() calls
  # example: grady_idx_3d(a, np.where(a < 10))
  # WARNING! no bounds checking here, ensure that idx is at least 1 away from each edge of array in y direction
  # 2nd order accurate grady at [i,j,k] = 3a[i+1,j+1,k] - 0.5a[i+1,j+2,k] - a[i+2,j+1,k]
  #                                       +a[i+2,j,k] - 2.5a[i+1,j,k]
  # WARNING! no bounds checking here, ensure that idx is at least 1 away from each edge of array in x direction
  grad_y = 3 * a[idx[0]+1, idx[1]+1, idx[2]] - 0.5 * a[idx[0]+1, idx[1]+2, idx[2]] - a[idx[0]+2, idx[1]+1, idx[2]] \
         + a[idx[0]+2, idx[1], idx[2]] - 2.5 * a[idx[0]+1, idx[1], idx[2]]
  return(grad_y)

def bottom_shiftright_idx_3d(a, idx):
  # compute gradient in y direction using a shifted x coord ala np.gradient for indices returned by np.where() calls
  # example: grady_idx_3d(a, np.where(a < 10))
  # WARNING! no bounds checking here, ensure that idx is at least 1 away from each edge of array in y direction
  grad_y = 3 * a[idx[0]-1, idx[1]+1, idx[2]] - 0.5 * a[idx[0]-1, idx[1]+2, idx[2]] - a[idx[0]-2, idx[1]+1, idx[2]] \
         + a[idx[0]-2, idx[1], idx[2]] - 2.5 * a[idx[0]-1, idx[1], idx[2]]
  return(grad_y)

def bottom_shiftrear_idx_3d(a, idx):
  # compute gradient in y direction using a shifted z coord ala np.gradient for indices returned by np.where() calls
  # example: grady_idx_3d(a, np.where(a < 10))
  # WARNING! no bounds checking here, ensure that idx is at least 1 away from each edge of array in y direction
  grad_y = 3 * a[idx[0], idx[1]+1, idx[2]+1] - 0.5 * a[idx[0], idx[1]+2, idx[2]+1] - a[idx[0], idx[1]+1, idx[2]+2] \
         + a[idx[0], idx[1], idx[2]+2] - 2.5 * a[idx[0], idx[1], idx[2]+1]
  return(grad_y)

def bottom_shiftfront_idx_3d(a, idx):
  # compute gradient in y direction using a shifted z coord ala np.gradient for indices returned by np.where() calls
  # example: grady_idx_3d(a, np.where(a < 10))
  # WARNING! no bounds checking here, ensure that idx is at least 1 away from each edge of array in y direction
  grad_y = 3 * a[idx[0], idx[1]+1, idx[2]-1] - 0.5 * a[idx[0], idx[1]+2, idx[2]-1] - a[idx[0], idx[1]+1, idx[2]-2] \
         + a[idx[0], idx[1], idx[2]-2] - 2.5 * a[idx[0], idx[1], idx[2]-1]
  return(grad_y)

def top_shiftleft_idx_3d(a, idx):
  # compute gradient in y direction using a shifted x coord ala np.gradient for indices returned by np.where() calls
  # example: grady_idx_3d(a, np.where(a < 10))
  # 2nd order accurate grady at [i,j,k] = -3a[i+1,j-1,k] + 0.5a[i+1,j-2,k] + a[i+2,j-1,k]
  #                                       -a[i+2,j,k] + 2.5a[i+1,j,k]
  # WARNING! no bounds checking here, ensure that idx is at least 1 away from each edge of array in x direction
  grad_y = -3 * a[idx[0]+1, idx[1]-1, idx[2]] + 0.5 * a[idx[0]+1, idx[1]-2, idx[2]] + a[idx[0]+2, idx[1]-1, idx[2]] \
         - a[idx[0]+2, idx[1], idx[2]] + 2.5 * a[idx[0]+1, idx[1], idx[2]]
  return(grad_y)

def top_shiftright_idx_3d(a, idx):
  # compute gradient in y direction using a shifted x coord ala np.gradient for indices returned by np.where() calls
  # example: grady_idx_3d(a, np.where(a < 10))
  # WARNING! no bounds checking here, ensure that idx is at least 1 away from each edge of array in y direction
  grad_y = -3 * a[idx[0]-1, idx[1]-1, idx[2]] + 0.5 * a[idx[0]-1, idx[1]-2, idx[2]] + a[idx[0]-2, idx[1]-1, idx[2]] \
         - a[idx[0]-2, idx[1], idx[2]] + 2.5 * a[idx[0]-1, idx[1], idx[2]]
  return(grad_y)

def top_shiftrear_idx_3d(a, idx):
  # compute gradient in y direction using a shifted z coord ala np.gradient for indices returned by np.where() calls
  # example: grady_idx_3d(a, np.where(a < 10))
  # WARNING! no bounds checking here, ensure that idx is at least 1 away from each edge of array in y direction
  grad_y = -3 * a[idx[0], idx[1]-1, idx[2]+1] + 0.5 * a[idx[0], idx[1]-2, idx[2]+1] + a[idx[0], idx[1]-1, idx[2]+2] \
         - a[idx[0], idx[1], idx[2]+2] + 2.5 * a[idx[0], idx[1], idx[2]+1]
  return(grad_y)

def top_shiftfront_idx_3d(a, idx):
  # compute gradient in y direction using a shifted z coord ala np.gradient for indices returned by np.where() calls
  # example: grady_idx_3d(a, np.where(a < 10))
  # WARNING! no bounds checking here, ensure that idx is at least 1 away from each edge of array in y direction
  grad_y = -3 * a[idx[0], idx[1]-1, idx[2]-1] + 0.5 * a[idx[0], idx[1]-2, idx[2]-1] + a[idx[0], idx[1]-1, idx[2]-2] \
         - a[idx[0], idx[1], idx[2]-2] + 2.5 * a[idx[0], idx[1], idx[2]-1]
  return(grad_y)

def gradz_shiftleft_idx_3d(a, idx):
  # compute gradient in z direction using a shifted x coord ala np.gradient for indices returned by np.where() calls
  # example: gradz_idx_3d(a, np.where(a < 10))
  # 2nd order accurate gradz at [i,j,k] = 0.5a[i+1,j,k+1]-0.5a[i+1,j,k-1]+0.25a[i+2,j,k+1]-0.25a[i+2,j,k-1]
  # WARNING! no bounds checking here, ensure that idx is at least 1 away from each edge of array in x direction
  grad_z = 0.5 * a[idx[0]+1, idx[1], idx[2]+1] - 0.5 * a[idx[0]+1, idx[1], idx[2]-1] \
         + 0.25 * a[idx[0]+2, idx[1], idx[2]+1] - 0.25 * a[idx[0]+2, idx[1], idx[2]-1]
  return(grad_z)

def gradz_shiftright_idx_3d(a, idx):
  # compute gradient in z direction using a shifted x coord ala np.gradient for indices returned by np.where() calls
  # example: gradz_idx_3d(a, np.where(a < 10))
  # WARNING! no bounds checking here, ensure that idx is at least 1 away from each edge of array in z direction
  grad_z = a[idx[0]-1, idx[1], idx[2]+1] - a[idx[0]-1, idx[1], idx[2]-1] \
         - 0.5 * a[idx[0]-2, idx[1], idx[2]+1] + 0.5 * a[idx[0]-2, idx[1], idx[2]-1]
  return(grad_z)

def gradz_shiftbottom_idx_3d(a, idx):
  # compute gradient in z direction using a shifted y coord ala np.gradient for indices returned by np.where() calls
  # example: gradz_idx_3d(a, np.where(a < 10))
  # WARNING! no bounds checking here, ensure that idx is at least 1 away from each edge of array in z direction
  grad_z = 0.5 * a[idx[0], idx[1]+1, idx[2]+1] - 0.5 * a[idx[0], idx[1]+1, idx[2]-1] \
         + 0.25 * a[idx[0], idx[1]+2, idx[2]+1] - 0.25 * a[idx[0], idx[1]+2, idx[2]-1]
  return(grad_z)

def gradz_shifttop_idx_3d(a, idx):
  # compute gradient in z direction using a shifted y coord ala np.gradient for indices returned by np.where() calls
  # example: gradz_idx_3d(a, np.where(a < 10))
  # WARNING! no bounds checking here, ensure that idx is at least 1 away from each edge of array in z direction
  grad_z = a[idx[0], idx[1]-1, idx[2]+1] - a[idx[0], idx[1]-1, idx[2]-1] \
         - 0.5 * a[idx[0], idx[1]-2, idx[2]+1] + 0.5 * a[idx[0], idx[1]-2, idx[2]-1]
  return(grad_z)

def rear_shiftleft_idx_3d(a, idx):
  # compute gradient in z direction using a shifted x coord ala np.gradient for indices returned by np.where() calls
  # example: gradz_idx_3d(a, np.where(a < 10))
  # 2nd order accurate gradz at [i,j,k] = 3a[i+1,j,k+1] - 0.5a[i+1,j,k+2] - a[i+2,j,k+1]
  #                                       +a[i+2,j,k] - 2.5a[i+1,j,k]
  # WARNING! no bounds checking here, ensure that idx is at least 1 away from each edge of array in x direction
  grad_z = 3 * a[idx[0]+1, idx[1], idx[2]+1] - 0.5 * a[idx[0]+1, idx[1], idx[2]+2] - a[idx[0]+2, idx[1], idx[2]+1] \
         + a[idx[0]+2, idx[1], idx[2]] - 2.5 * a[idx[0]+1, idx[1], idx[2]]
  return(grad_z)

def rear_shiftright_idx_3d(a, idx):
  # compute gradient in z direction using a shifted x coord ala np.gradient for indices returned by np.where() calls
  # example: gradz_idx_3d(a, np.where(a < 10))
  # WARNING! no bounds checking here, ensure that idx is at least 1 away from each edge of array in z direction
  grad_z = 3 * a[idx[0]-1, idx[1], idx[2]+1] - 0.5 * a[idx[0]-1, idx[1], idx[2]+2] - a[idx[0]-2, idx[1], idx[2]+1] \
         + a[idx[0]-2, idx[1], idx[2]] - 2.5 * a[idx[0]-1, idx[1], idx[2]]
  return(grad_z)

def rear_shiftbottom_idx_3d(a, idx):
  # compute gradient in z direction using a shifted y coord ala np.gradient for indices returned by np.where() calls
  # example: gradz_idx_3d(a, np.where(a < 10))
  # WARNING! no bounds checking here, ensure that idx is at least 1 away from each edge of array in z direction
  grad_z = 3 * a[idx[0], idx[1]+1, idx[2]+1] - 0.5 * a[idx[0], idx[1]+1, idx[2]+2] - a[idx[0], idx[1]+2, idx[2]+1] \
         + a[idx[0], idx[1]+2, idx[2]] - 2.5 * a[idx[0], idx[1]+1, idx[2]]
  return(grad_z)

def rear_shifttop_idx_3d(a, idx):
  # compute gradient in z direction using a shifted y coord ala np.gradient for indices returned by np.where() calls
  # example: gradz_idx_3d(a, np.where(a < 10))
  # WARNING! no bounds checking here, ensure that idx is at least 1 away from each edge of array in z direction
  grad_z = 3 * a[idx[0], idx[1]-1, idx[2]+1] - 0.5 * a[idx[0], idx[1]-1, idx[2]+2] - a[idx[0], idx[1]-2, idx[2]+1] \
         + a[idx[0], idx[1]-2, idx[2]] - 2.5 * a[idx[0], idx[1]-1, idx[2]]
  return(grad_z)

def front_shiftleft_idx_3d(a, idx):
  # compute gradient in z direction using a shifted x coord ala np.gradient for indices returned by np.where() calls
  # example: gradz_idx_3d(a, np.where(a < 10))
  # 2nd order accurate gradz at [i,j,k] = -3a[i+1,j,k-1] + 0.5a[i+1,j,k-2] + a[i+2,j,k-1]
  #                                       -a[i+2,j,k] + 2.5a[i+1,j,k]
  # WARNING! no bounds checking here, ensure that idx is at least 1 away from each edge of array in x direction
  grad_z = -3 * a[idx[0]+1, idx[1], idx[2]-1] + 0.5 * a[idx[0]+1, idx[1], idx[2]-2] + a[idx[0]+2, idx[1], idx[2]-1] \
         - a[idx[0]+2, idx[1], idx[2]] + 2.5 * a[idx[0]+1, idx[1], idx[2]]
  return(grad_z)

def front_shiftright_idx_3d(a, idx):
  # compute gradient in z direction using a shifted x coord ala np.gradient for indices returned by np.where() calls
  # example: gradz_idx_3d(a, np.where(a < 10))
  # WARNING! no bounds checking here, ensure that idx is at least 1 away from each edge of array in z direction
  grad_z = -3 * a[idx[0]-1, idx[1], idx[2]-1] + 0.5 * a[idx[0]-1, idx[1], idx[2]-2] + a[idx[0]-2, idx[1], idx[2]-1] \
         - a[idx[0]-2, idx[1], idx[2]] + 2.5 * a[idx[0]-1, idx[1], idx[2]]
  return(grad_z)

def front_shiftbottom_idx_3d(a, idx):
  # compute gradient in z direction using a shifted y coord ala np.gradient for indices returned by np.where() calls
  # example: gradz_idx_3d(a, np.where(a < 10))
  # WARNING! no bounds checking here, ensure that idx is at least 1 away from each edge of array in z direction
  grad_z = -3 * a[idx[0], idx[1]+1, idx[2]-1] + 0.5 * a[idx[0], idx[1]+1, idx[2]-2] + a[idx[0], idx[1]+2, idx[2]-1] \
         - a[idx[0], idx[1]+2, idx[2]] + 2.5 * a[idx[0], idx[1]+1, idx[2]]
  return(grad_z)

def front_shifttop_idx_3d(a, idx):
  # compute gradient in z direction using a shifted y coord ala np.gradient for indices returned by np.where() calls
  # example: gradz_idx_3d(a, np.where(a < 10))
  # WARNING! no bounds checking here, ensure that idx is at least 1 away from each edge of array in z direction
  grad_z = -3 * a[idx[0], idx[1]-1, idx[2]-1] + 0.5 * a[idx[0], idx[1]-1, idx[2]-2] + a[idx[0], idx[1]-2, idx[2]-1] \
         - a[idx[0], idx[1]-2, idx[2]] + 2.5 * a[idx[0], idx[1]-1, idx[2]]
  return(grad_z)

def left_diff_3d(a):
  # return the derivative in the x direction using only terms to the right
  # called left, because used for case when left direction (i-1) is out of bounds
  left_x = np.zeros_like(a)
  left_x[:-2,:,:] = -0.5 * a[2:,:,:] + 2.0 * a[1:-1,:,:] - 1.5 * a[:-2,:,:]
  return(left_x)

def left_diff_idx_3d(a, idx):
  # return the derivative in the x direction using only terms to the right
  # called left, because used for case when left direction (i-1) is out of bounds
  # compute for indices returned by np.where() calls
  # example: left_diff_idx_3d(a, np.where(a < 10))
  # WARNING! no bounds checking here, ensure that idx[0] is at least 3 less than array size in x direction
  left_x = -0.5 * a[idx[0]+2, idx[1], idx[2]] + 2.0 * a[idx[0]+1, idx[1], idx[2]] - 1.5 * a[idx[0], idx[1], idx[2]]
  return(left_x)

def right_diff_3d(a):
  # return the derivative in the x direction using only terms to the left
  # called right, because used for case when right direction (i+1) is out of bounds
  right_x = np.zeros_like(a)
  right_x[2:,:,:] = 0.5 * a[:-2,:,:] - 2.0 * a[1:-1,:,:] + 1.5 * a[2:,:,:]
  return(right_x)

def right_diff_idx_3d(a, idx):
  # return the derivative in the x direction using only terms to the left
  # called right, because used for case when right direction (i+1) is out of bounds
  # compute for indices returned by np.where() calls
  # example: right_diff_idx_3d(a, np.where(a < 10))
  # WARNING! no bounds checking here, ensure that idx[0] is at least 2 in the x direction
  right_x = 0.5 * a[idx[0]-2, idx[1], idx[2]] - 2.0 * a[idx[0]-1, idx[1], idx[2]] + 1.5 * a[idx[0], idx[1], idx[2]]
  return(right_x)
  
def bottom_diff_3d(a):
  # return the derivative in the y direction using only terms to the top (increasing y direction)
  # called bottom, because used for case when bottom direction (j-1) is out of bounds
  bot_x = np.zeros_like(a)
  bot_x[:,:-2,:] = -0.5 * a[:,2:,:] + 2.0 * a[:,1:-1,:] - 1.5 * a[:,:-2,:]
  return(bot_x)

def bottom_diff_idx_3d(a, idx):
  # return the derivative in the y direction using only terms to the top (increasing y direction)
  # called bottom, because used for case when bottom direction (j-1) is out of bounds
  # compute for indices returned by np.where() calls
  # example: bottom_diff_idx_3d(a, np.where(a < 10))
  # WARNING! no bounds checking here, ensure that idx[1] is at least 3 less than array size in y direction
  bot_x = -0.5 * a[idx[0], idx[1]+2, idx[2]] + 2.0 * a[idx[0],idx[1]+1, idx[2]] - 1.5 * a[idx[0],idx[1], idx[2]]
  return(bot_x)

def top_diff_3d(a):
  # return the derivative in the y direction using only terms to the bottom (decreasing y direction)
  # called top, because used for case when top direction (j+1) is out of bounds
  top_x = np.zeros_like(a)
  top_x[:,2:,:] = 0.5 * a[:,:-2,:] - 2.0 * a[:,1:-1,:] + 1.5 * a[:,2:,:]
  return(top_x)

def top_diff_idx_3d(a, idx):
  # return the derivative in the y direction using only terms to the bottom (decreasing y direction)
  # called top, because used for case when top direction (j+1) is out of bounds
  # compute for indices returned by np.where() calls
  # example: top_diff_idx_3d(a, np.where(a < 10))
  # WARNING! no bounds checking here, ensure that idx[1] is at least 2 in y direction
  top_x = 0.5 * a[idx[0], idx[1]-2, idx[2]] - 2.0 * a[idx[0], idx[1]-1, idx[2]] + 1.5 * a[idx[0], idx[1], idx[2]]
  return(top_x)

def rear_diff_3d(a):
  # return the derivative in the z direction using only terms to the front (increasing z direction)
  # called rear, because used for case when rear direction (k-1) is out of bounds
  rear_x = np.zeros_like(a)
  rear_x[:,:,:-2] = -0.5 * a[:,:,2:] + 2.0 * a[:,:,1:-1] - 1.5 * a[:,:,:-2]
  return(rear_x)

def rear_diff_idx_3d(a, idx):
  # return the derivative in the zy direction using only terms to the front (increasing z direction)
  # called rear, because used for case when rear direction (l-1) is out of bounds
  # compute for indices returned by np.where() calls
  # example: rear_diff_idx_3d(a, np.where(a < 10))
  # WARNING! no bounds checking here, ensure that idx[1] is at least 3 less than array size in z direction
  rear_x = -0.5 * a[idx[0], idx[1], idx[2]+2] + 2.0 * a[idx[0],idx[1], idx[2]+1] - 1.5 * a[idx[0],idx[1], idx[2]]
  return(rear_x)

def front_diff_3d(a):
  # return the derivative in the z direction using only terms to the rear (decreasing z direction)
  # called front, because used for case when front direction (k+1) is out of bounds
  front_x = np.zeros_like(a)
  front_x[:,:,2:] = 0.5 * a[:,:,:-2] - 2.0 * a[:,:,1:-1] + 1.5 * a[:,:,2:]
  return(front_x)

def front_diff_idx_3d(a, idx):
  # return the derivative in the z direction using only terms to the rear (decreasing z direction)
  # called front, because used for case when front direction (k+1) is out of bounds
  # compute for indices returned by np.where() calls
  # example: front_diff_idx_3d(a, np.where(a < 10))
  # WARNING! no bounds checking here, ensure that idx[1] is at least 2 in z direction
  front_x = 0.5 * a[idx[0], idx[1], idx[2]-2] - 2.0 * a[idx[0], idx[1], idx[2]-1] + 1.5 * a[idx[0], idx[1], idx[2]]
  return(front_x)

############################################################
# Find gradient of image inside mask region                #
# Using appropriate left,right,top,bottom,front,rear diffs #
# at the boundary                                          #
############################################################
def gradient_mask_3d(img, mask, modify_mask=True):
  # Find the gradient of img inside the mask regions
  # Assumes it's a scalar 3D image
  # Returns grad_x, grad_y, grad_z
  # If modify_mask is False, this method assumes that mask already consists of only differentiable points
  # Otherwise it will mask out nondifferentiable points
  
  # We call this twice because first time modifies the mask
  # TODO see if we can handle the nondifferentiable boundary types with more grace
  # Its really that we need a second order scheme for really small structures.
  if modify_mask:
    bdry_type, bdry_idx, bdry_map = mo.determine_boundary_3d(mask, True)
  bdry_type, bdry_idx, bdry_map = mo.determine_boundary_3d(mask, False)

  gradX, gradY, gradZ = gradient_bdry_3d(img, bdry_idx, bdry_map)
  return(gradX, gradY, gradZ)
# def end gradient_mask_3d

def gradient_bdry_3d(img, bdry_idx, bdry_map):

  xsz = img.shape[0]
  ysz = img.shape[1]
  zsz = img.shape[2]
  gradX = np.zeros((xsz,ysz,zsz))
  gradY = np.zeros((xsz,ysz,zsz))
  gradZ = np.zeros((xsz,ysz,zsz))

  for btype, bnum in bdry_map.items():
    if bnum == 0:
      # skip, since outside
      continue

    b_idx = bdry_idx[bnum]

    xdiff,ydiff,zdiff = select_diffs_3d(btype,zero_shifts=False)

    if btype == "outside":
      # outside mask, skip
      pass
    else:
      gradX[b_idx[0], b_idx[1], b_idx[2]] = xdiff(img, b_idx)
      gradY[b_idx[0], b_idx[1], b_idx[2]] = ydiff(img, b_idx)
      gradZ[b_idx[0], b_idx[1], b_idx[2]] = zdiff(img, b_idx)
    
    # if btype[0:8] == "interior":
    #   gradX[b_idx[0], b_idx[1], b_idx[2]] = gradx_idx_3d(img, b_idx)
    #   gradY[b_idx[0], b_idx[1], b_idx[2]] = grady_idx_3d(img, b_idx)
    #   gradZ[b_idx[0], b_idx[1], b_idx[2]] = gradz_idx_3d(img, b_idx)
      
    # elif btype == "left":
    #   gradX[b_idx[0], b_idx[1], b_idx[2]] = left_diff_idx_3d(img, b_idx)
    #   gradY[b_idx[0], b_idx[1], b_idx[2]] = grady_idx_3d(img, b_idx)
    #   gradZ[b_idx[0], b_idx[1], b_idx[2]] = gradz_idx_3d(img, b_idx)

    # elif btype == "bottomleft":
    #   gradX[b_idx[0], b_idx[1], b_idx[2]] = left_diff_idx_3d(img, b_idx)
    #   gradY[b_idx[0], b_idx[1], b_idx[2]] = bottom_diff_idx_3d(img, b_idx)
    #   gradZ[b_idx[0], b_idx[1], b_idx[2]] = gradz_idx_3d(img, b_idx)

    # elif btype == "topleft":
    #   gradX[b_idx[0], b_idx[1], b_idx[2]] = left_diff_idx_3d(img, b_idx)
    #   gradY[b_idx[0], b_idx[1], b_idx[2]] = top_diff_idx_3d(img, b_idx)
    #   gradZ[b_idx[0], b_idx[1], b_idx[2]] = gradz_idx_3d(img, b_idx)

    # elif btype == "notright":
    #   gradX[b_idx[0], b_idx[1], b_idx[2]] = left_diff_idx_3d(img, b_idx)
    #   gradY[b_idx[0], b_idx[1], b_idx[2]] = grady_idx_3d(img, [b_idx[0]+1, b_idx[1], b_idx[2]])
    #   gradZ[b_idx[0], b_idx[1], b_idx[2]] = gradz_idx_3d(img, [b_idx[0]+1, b_idx[1], b_idx[2]])

    # elif btype == "right":
    #   gradX[b_idx[0], b_idx[1], b_idx[2]] = right_diff_idx_3d(img, b_idx)
    #   gradY[b_idx[0], b_idx[1], b_idx[2]] = grady_idx_3d(img, b_idx)
    #   gradZ[b_idx[0], b_idx[1], b_idx[2]] = gradz_idx_3d(img, b_idx)

    # elif btype == "bottomright":
    #   gradX[b_idx[0], b_idx[1], b_idx[2]] = right_diff_idx_3d(img, b_idx)
    #   gradY[b_idx[0], b_idx[1], b_idx[2]] = bottom_diff_idx_3d(img, b_idx)
    #   gradZ[b_idx[0], b_idx[1], b_idx[2]] = gradz_idx_3d(img, b_idx)

    # elif btype == "topright":
    #   gradX[b_idx[0], b_idx[1], b_idx[2]] = right_diff_idx_3d(img, b_idx)
    #   gradY[b_idx[0], b_idx[1], b_idx[2]] = top_diff_idx_3d(img, b_idx)
    #   gradZ[b_idx[0], b_idx[1], b_idx[2]] = gradz_idx_3d(img, b_idx)

    # elif btype == "notleft":
    #   gradX[b_idx[0], b_idx[1], b_idx[2]] = right_diff_idx_3d(img, b_idx)
    #   gradY[b_idx[0], b_idx[1], b_idx[2]] = grady_idx_3d(img, [b_idx[0]-1, b_idx[1], b_idx[2]])
    #   gradZ[b_idx[0], b_idx[1], b_idx[2]] = gradz_idx_3d(img, [b_idx[0]-1, b_idx[1], b_idx[2]])

    # elif btype == "bottom":
    #   gradX[b_idx[0], b_idx[1], b_idx[2]] = gradx_idx_3d(img, b_idx)
    #   gradY[b_idx[0], b_idx[1], b_idx[2]] = bottom_diff_idx_3d(img, b_idx)
    #   gradZ[b_idx[0], b_idx[1], b_idx[2]] = gradz_idx_3d(img, b_idx)

    # elif btype == "nottop":
    #   gradX[b_idx[0], b_idx[1], b_idx[2]] = gradx_idx_3d(img, [b_idx[0], b_idx[1]+1, b_idx[2]])
    #   gradY[b_idx[0], b_idx[1], b_idx[2]] = bottom_diff_idx_3d(img, b_idx)
    #   gradZ[b_idx[0], b_idx[1], b_idx[2]] = gradz_idx_3d(img, [b_idx[0], b_idx[1]+1, b_idx[2]])
      
    # elif btype == "top":
    #   gradX[b_idx[0], b_idx[1], b_idx[2]] = gradx_idx_3d(img, b_idx)
    #   gradY[b_idx[0], b_idx[1], b_idx[2]] = top_diff_idx_3d(img, b_idx)
    #   gradZ[b_idx[0], b_idx[1], b_idx[2]] = gradz_idx_3d(img, b_idx)

    # elif btype == "notbottom":
    #   gradX[b_idx[0], b_idx[1], b_idx[2]] = gradx_idx_3d(img, [b_idx[0], b_idx[1]-1, b_idx[2]])
    #   gradY[b_idx[0], b_idx[1], b_idx[2]] = top_diff_idx_3d(img, b_idx)
    #   gradZ[b_idx[0], b_idx[1], b_idx[2]] = gradz_idx_3d(img, [b_idx[0], b_idx[1]-1, b_idx[2]])

    # elif btype == "rear":
    #   gradX[b_idx[0], b_idx[1], b_idx[2]] = gradx_idx_3d(img, b_idx)
    #   gradY[b_idx[0], b_idx[1], b_idx[2]] = grady_idx_3d(img, b_idx)
    #   gradZ[b_idx[0], b_idx[1], b_idx[2]] = rear_diff_idx_3d(img, b_idx)

    # elif btype == "rearleft":
    #   gradX[b_idx[0], b_idx[1], b_idx[2]] = left_diff_idx_3d(img, b_idx)
    #   gradY[b_idx[0], b_idx[1], b_idx[2]] = grady_idx_3d(img, b_idx)
    #   gradZ[b_idx[0], b_idx[1], b_idx[2]] = rear_diff_idx_3d(img, b_idx)

    # elif btype == "rearright":
    #   gradX[b_idx[0], b_idx[1], b_idx[2]] = right_diff_idx_3d(img, b_idx)
    #   gradY[b_idx[0], b_idx[1], b_idx[2]] = grady_idx_3d(img, b_idx)
    #   gradZ[b_idx[0], b_idx[1], b_idx[2]] = rear_diff_idx_3d(img, b_idx)

    # elif btype == "rearbottom":
    #   gradX[b_idx[0], b_idx[1], b_idx[2]] = gradx_idx_3d(img, b_idx)
    #   gradY[b_idx[0], b_idx[1], b_idx[2]] = bottom_diff_idx_3d(img, b_idx)
    #   gradZ[b_idx[0], b_idx[1], b_idx[2]] = rear_diff_idx_3d(img, b_idx)

    # elif btype == "reartop":
    #   gradX[b_idx[0], b_idx[1], b_idx[2]] = gradx_idx_3d(img, b_idx)
    #   gradY[b_idx[0], b_idx[1], b_idx[2]] = top_diff_idx_3d(img, b_idx)
    #   gradZ[b_idx[0], b_idx[1], b_idx[2]] = rear_diff_idx_3d(img, b_idx)

    # elif btype == "rearbottomleft":
    #   gradX[b_idx[0], b_idx[1], b_idx[2]] = left_diff_idx_3d(img, b_idx)
    #   gradY[b_idx[0], b_idx[1], b_idx[2]] = bottom_diff_idx_3d(img, b_idx)
    #   gradZ[b_idx[0], b_idx[1], b_idx[2]] = rear_diff_idx_3d(img, b_idx)

    # elif btype == "reartopleft":
    #   gradX[b_idx[0], b_idx[1], b_idx[2]] = left_diff_idx_3d(img, b_idx)
    #   gradY[b_idx[0], b_idx[1], b_idx[2]] = top_diff_idx_3d(img, b_idx)
    #   gradZ[b_idx[0], b_idx[1], b_idx[2]] = rear_diff_idx_3d(img, b_idx)
  
    # elif btype == "rearbottomright":
    #   gradX[b_idx[0], b_idx[1], b_idx[2]] = right_diff_idx_3d(img, b_idx)
    #   gradY[b_idx[0], b_idx[1], b_idx[2]] = bottom_diff_idx_3d(img, b_idx)
    #   gradZ[b_idx[0], b_idx[1], b_idx[2]] = rear_diff_idx_3d(img, b_idx)

    # elif btype == "reartopright":
    #   gradX[b_idx[0], b_idx[1], b_idx[2]] = right_diff_idx_3d(img, b_idx)
    #   gradY[b_idx[0], b_idx[1], b_idx[2]] = top_diff_idx_3d(img, b_idx)
    #   gradZ[b_idx[0], b_idx[1], b_idx[2]] = rear_diff_idx_3d(img, b_idx)

    # elif btype == "notfront":
    #   gradX[b_idx[0], b_idx[1], b_idx[2]] = gradx_idx_3d(img, [b_idx[0], b_idx[1], b_idx[2]+1])
    #   gradY[b_idx[0], b_idx[1], b_idx[2]] = grady_idx_3d(img, [b_idx[0], b_idx[1], b_idx[2]+1])
    #   gradZ[b_idx[0], b_idx[1], b_idx[2]] = rear_diff_idx_3d(img, b_idx)

    # elif btype == "front":
    #   gradX[b_idx[0], b_idx[1], b_idx[2]] = gradx_idx_3d(img, b_idx)
    #   gradY[b_idx[0], b_idx[1], b_idx[2]] = grady_idx_3d(img, b_idx)
    #   gradZ[b_idx[0], b_idx[1], b_idx[2]] = front_diff_idx_3d(img, b_idx)

    # elif btype == "frontleft":
    #   gradX[b_idx[0], b_idx[1], b_idx[2]] = left_diff_idx_3d(img, b_idx)
    #   gradY[b_idx[0], b_idx[1], b_idx[2]] = grady_idx_3d(img, b_idx)
    #   gradZ[b_idx[0], b_idx[1], b_idx[2]] = front_diff_idx_3d(img, b_idx)

    # elif btype == "frontright":
    #   gradX[b_idx[0], b_idx[1], b_idx[2]] = right_diff_idx_3d(img, b_idx)
    #   gradY[b_idx[0], b_idx[1], b_idx[2]] = grady_idx_3d(img, b_idx)
    #   gradZ[b_idx[0], b_idx[1], b_idx[2]] = front_diff_idx_3d(img, b_idx)

    # elif btype == "frontbottom":
    #   gradX[b_idx[0], b_idx[1], b_idx[2]] = gradx_idx_3d(img, b_idx)
    #   gradY[b_idx[0], b_idx[1], b_idx[2]] = bottom_diff_idx_3d(img, b_idx)
    #   gradZ[b_idx[0], b_idx[1], b_idx[2]] = front_diff_idx_3d(img, b_idx)

    # elif btype == "fronttop":
    #   gradX[b_idx[0], b_idx[1], b_idx[2]] = gradx_idx_3d(img, b_idx)
    #   gradY[b_idx[0], b_idx[1], b_idx[2]] = top_diff_idx_3d(img, b_idx)
    #   gradZ[b_idx[0], b_idx[1], b_idx[2]] = front_diff_idx_3d(img, b_idx)

    # elif btype == "frontbottomleft":
    #   gradX[b_idx[0], b_idx[1], b_idx[2]] = left_diff_idx_3d(img, b_idx)
    #   gradY[b_idx[0], b_idx[1], b_idx[2]] = bottom_diff_idx_3d(img, b_idx)
    #   gradZ[b_idx[0], b_idx[1], b_idx[2]] = front_diff_idx_3d(img, b_idx)

    # elif btype == "fronttopleft":
    #   gradX[b_idx[0], b_idx[1], b_idx[2]] = left_diff_idx_3d(img, b_idx)
    #   gradY[b_idx[0], b_idx[1], b_idx[2]] = top_diff_idx_3d(img, b_idx)
    #   gradZ[b_idx[0], b_idx[1], b_idx[2]] = front_diff_idx_3d(img, b_idx)
      
    # elif btype == "frontbottomright":
    #   gradX[b_idx[0], b_idx[1], b_idx[2]] = right_diff_idx_3d(img, b_idx)
    #   gradY[b_idx[0], b_idx[1], b_idx[2]] = bottom_diff_idx_3d(img, b_idx)
    #   gradZ[b_idx[0], b_idx[1], b_idx[2]] = front_diff_idx_3d(img, b_idx)

    # elif btype == "fronttopright":
    #   gradX[b_idx[0], b_idx[1], b_idx[2]] = right_diff_idx_3d(img, b_idx)
    #   gradY[b_idx[0], b_idx[1], b_idx[2]] = top_diff_idx_3d(img, b_idx)
    #   gradZ[b_idx[0], b_idx[1], b_idx[2]] = front_diff_idx_3d(img, b_idx)

    # elif btype == "notrear":
    #   gradX[b_idx[0], b_idx[1], b_idx[2]] = gradx_idx_3d(img, [b_idx[0], b_idx[1], b_idx[2]-1])
    #   gradY[b_idx[0], b_idx[1], b_idx[2]] = grady_idx_3d(img, [b_idx[0], b_idx[1], b_idx[2]-1])
    #   gradZ[b_idx[0], b_idx[1], b_idx[2]] = front_diff_idx_3d(img, b_idx)

    # elif btype == "outside":
    #   # outside mask, skip
    #   pass

    # else:
    #   # unrecognized type
    #   print(btype, "unrecognized.  Skipping")
      
  return(gradX, gradY, gradZ)
# end gradient_mask_3d  

################################
# Begin eigenvector diff opers #
################################
# Following 3D operators are for derivatives of eigenvector fields
# The specialized code here is to handle the natural ambiguities in eigenvector direction (v vs -v)

def eigv_gradx_idx_3d(eigvecs, idx):
  # Return the gradient of each component of the eigenvector taken in the x direction
  gradx_delx = np.zeros(len(idx[0]))
  grady_delx = np.zeros(len(idx[0]))
  gradz_delx = np.zeros(len(idx[0]))
  
  for ii in range(len(idx[0])):
    xx = idx[0][ii]
    yy = idx[1][ii]
    zz = idx[2][ii]
    pix = eigvecs[xx,yy,zz]
    left = eigvecs[xx-1,yy,zz]
    right = eigvecs[xx+1,yy,zz]

    if np.dot(right, left) < 0:
      if np.dot(right, pix) < 0:
        right = -right
      else:
        left = -left
    #if np.dot(pix, right) < 0:
    #  right = -right

    gradx_delx[ii] = 0.5 * (right[0] - left[0])
    grady_delx[ii] = 0.5 * (right[1] - left[1])
    gradz_delx[ii] = 0.5 * (right[2] - left[2])
  return(gradx_delx, grady_delx, gradz_delx)

def eigv_grady_idx_3d(eigvecs, idx):
  # Return the gradient of each component of the eigenvector taken in the y direction
  gradx_dely = np.zeros(len(idx[0]))
  grady_dely = np.zeros(len(idx[0]))
  gradz_dely = np.zeros(len(idx[0]))
  
  for ii in range(len(idx[0])):
    xx = idx[0][ii]
    yy = idx[1][ii]
    zz = idx[2][ii]
    pix = eigvecs[xx,yy,zz]
    bottom = eigvecs[xx,yy-1,zz]
    top = eigvecs[xx,yy+1,zz]

    if np.dot(top, bottom) < 0:
      if np.dot(top, pix) < 0:
        top = -top
      else:
        bottom = -bottom
      
    #if np.dot(pix, top) < 0:
    #  top = -top

    gradx_dely[ii] = 0.5 * (top[0] - bottom[0])
    grady_dely[ii] = 0.5 * (top[1] - bottom[1])
    gradz_dely[ii] = 0.5 * (top[2] - bottom[2])
  return(gradx_dely, grady_dely, gradz_dely)

def eigv_gradz_idx_3d(eigvecs, idx):
  # Return the gradient of each component of the eigenvector taken in the z direction
  gradx_delz = np.zeros(len(idx[0]))
  grady_delz = np.zeros(len(idx[0]))
  gradz_delz = np.zeros(len(idx[0]))
  
  for ii in range(len(idx[0])):
    xx = idx[0][ii]
    yy = idx[1][ii]
    zz = idx[2][ii]
    pix = eigvecs[xx,yy,zz]
    rear = eigvecs[xx,yy,zz-1]
    front = eigvecs[xx,yy,zz+1]

    if np.dot(front, rear) < 0:
      if np.dot(front, pix) < 0:
        front = -front
      else:
        rear = -rear

    gradx_delz[ii] = 0.5 * (front[0] - rear[0])
    grady_delz[ii] = 0.5 * (front[1] - rear[1])
    gradz_delz[ii] = 0.5 * (front[2] - rear[2])
  return(gradx_delz, grady_delz, gradz_delz)

def eigv_gradx_shiftbottom_idx_3d(eigvecs, idx):
  # Return the gradient of each component of the eigenvector taken in the x direction, shifted in the y direction
  gradx_delx = np.zeros(len(idx[0]))
  grady_delx = np.zeros(len(idx[0]))
  gradz_delx = np.zeros(len(idx[0]))
  
  for ii in range(len(idx[0])):
    xx = idx[0][ii]
    yy = idx[1][ii]
    zz = idx[2][ii]
    pix = eigvecs[xx,yy+1,zz]
    left = eigvecs[xx-1,yy+1,zz]
    right = eigvecs[xx+1,yy+1,zz]
    left2 = eigvecs[xx-1,yy+2,zz]
    right2 = eigvecs[xx+1,yy+2,zz]

    if np.dot(right, left) < 0:
      if np.dot(right, pix) < 0:
        right = -right
      else:
        left = -left

    # 2nd order accurate gradx at [i,j,k] = 0.5a[i+1,j+1,k]-0.5a[i-1,j+1,k]+0.25a[i+1,j+2,k]-0.25a[i-1,j+2,k]
    gradx_delx[ii] = 0.5 * right[0] - 0.5 * left[0] + 0.25 * right2[0] -0.25 * left2[0]
    grady_delx[ii] = 0.5 * right[1] - 0.5 * left[1] + 0.25 * right2[1] -0.25 * left2[1]
    gradz_delx[ii] = 0.5 * right[2] - 0.5 * left[2] + 0.25 * right2[2] -0.25 * left2[2]
  return(gradx_delx, grady_delx, gradz_delx)

def eigv_gradx_shifttop_idx_3d(eigvecs, idx):
  # Return the gradient of each component of the eigenvector taken in the x direction, shifted in the y direction
  gradx_delx = np.zeros(len(idx[0]))
  grady_delx = np.zeros(len(idx[0]))
  gradz_delx = np.zeros(len(idx[0]))
  
  for ii in range(len(idx[0])):
    xx = idx[0][ii]
    yy = idx[1][ii]
    zz = idx[2][ii]
    pix = eigvecs[xx,yy-1,zz]
    left = eigvecs[xx-1,yy-1,zz]
    right = eigvecs[xx+1,yy-1,zz]
    left2 = eigvecs[xx-1,yy-2,zz]
    right2 = eigvecs[xx+1,yy-2,zz]

    if np.dot(right, left) < 0:
      if np.dot(right, pix) < 0:
        right = -right
      else:
        left = -left
        
    # 2nd order accurate gradx at [i,j,k] = a[i+1,j-1,k]-a[i-1,j-1,k]-0.5a[i+1,j-2,k]+0.5a[i-1,j-2,k]
    gradx_delx[ii] = right[0] - left[0] - 0.5 * right2[0] + 0.5 * left2[0]
    grady_delx[ii] = right[1] - left[1] - 0.5 * right2[1] + 0.5 * left2[1]
    gradz_delx[ii] = right[2] - left[2] - 0.5 * right2[2] + 0.5 * left2[2]
  return(gradx_delx, grady_delx, gradz_delx)

def eigv_gradx_shiftrear_idx_3d(eigvecs, idx):
  # Return the gradient of each component of the eigenvector taken in the x direction, shifted in the z direction
  gradx_delx = np.zeros(len(idx[0]))
  grady_delx = np.zeros(len(idx[0]))
  gradz_delx = np.zeros(len(idx[0]))
  
  for ii in range(len(idx[0])):
    xx = idx[0][ii]
    yy = idx[1][ii]
    zz = idx[2][ii]
    pix = eigvecs[xx,yy,zz+1]
    left = eigvecs[xx-1,yy,zz+1]
    right = eigvecs[xx+1,yy,zz+1]
    left2 = eigvecs[xx-1,yy,zz+2]
    right2 = eigvecs[xx+1,yy,zz+2]

    if np.dot(right, left) < 0:
      if np.dot(right, pix) < 0:
        right = -right
      else:
        left = -left

    gradx_delx[ii] = 0.5 * right[0] - 0.5 * left[0] + 0.25 * right2[0] -0.25 * left2[0]
    grady_delx[ii] = 0.5 * right[1] - 0.5 * left[1] + 0.25 * right2[1] -0.25 * left2[1]
    gradz_delx[ii] = 0.5 * right[2] - 0.5 * left[2] + 0.25 * right2[2] -0.25 * left2[2]
  return(gradx_delx, grady_delx, gradz_delx)

def eigv_gradx_shiftfront_idx_3d(eigvecs, idx):
  # Return the gradient of each component of the eigenvector taken in the x direction, shifted in the z direction
  gradx_delx = np.zeros(len(idx[0]))
  grady_delx = np.zeros(len(idx[0]))
  gradz_delx = np.zeros(len(idx[0]))
  
  for ii in range(len(idx[0])):
    xx = idx[0][ii]
    yy = idx[1][ii]
    zz = idx[2][ii]
    pix = eigvecs[xx,yy,zz-1]
    left = eigvecs[xx-1,yy,zz-1]
    right = eigvecs[xx+1,yy,zz-1]
    left2 = eigvecs[xx-1,yy,zz-2]
    right2 = eigvecs[xx+1,yy,zz-2]

    if np.dot(right, left) < 0:
      if np.dot(right, pix) < 0:
        right = -right
      else:
        left = -left

    # 2nd order accurate gradx at [i,j,k] = a[i+1,j-1,k]-a[i-1,j-1,k]-0.5a[i+1,j-2,k]+0.5a[i-1,j-2,k]
    gradx_delx[ii] = right[0] - left[0] - 0.5 * right2[0] + 0.5 * left2[0]
    grady_delx[ii] = right[1] - left[1] - 0.5 * right2[1] + 0.5 * left2[1]
    gradz_delx[ii] = right[2] - left[2] - 0.5 * right2[2] + 0.5 * left2[2]
  return(gradx_delx, grady_delx, gradz_delx)

def eigv_left_shiftbottom_idx_3d(eigvecs, idx):
  # Return the gradient of each component of the eigenvector taken in the x direction, shifted in the y direction
  gradx_delx = np.zeros(len(idx[0]))
  grady_delx = np.zeros(len(idx[0]))
  gradz_delx = np.zeros(len(idx[0]))
  
  for ii in range(len(idx[0])):
    xx = idx[0][ii]
    yy = idx[1][ii]
    zz = idx[2][ii]
    pix = eigvecs[xx,yy+1,zz]
    right = eigvecs[xx+1,yy+1,zz]
    right2 = eigvecs[xx+2,yy+1,zz]
    top2 = eigvecs[xx,yy+2,zz]
    righttop2 = eigvecs[xx+1,yy+2,zz]
    
    if np.dot(pix, right) < 0:
      right = -right
    if np.dot(pix, right2) < 0:
      right2 = -right2

    # 2nd order accurate gradx at [i,j,k] = 3a[i+1,j+1,k] - 0.5a[i+2,j+1,k] - a[i+1,j+2,k]
    #                                       +a[i,j+2,k] - 2.5a[i,j+1,k]
    gradx_delx[ii] = 3*right[0] - 0.5*right2[0] -righttop2[0] + top2[0] - 2.5*pix[0]
    grady_delx[ii] = 3*right[1] - 0.5*right2[1] -righttop2[1] + top2[1] - 2.5*pix[1]
    gradz_delx[ii] = 3*right[2] - 0.5*right2[2] -righttop2[2] + top2[2] - 2.5*pix[2]
  return(gradx_delx, grady_delx, gradz_delx)

def eigv_left_shifttop_idx_3d(eigvecs, idx):
  # Return the gradient of each component of the eigenvector taken in the x direction, shifted in the y direction
  gradx_delx = np.zeros(len(idx[0]))
  grady_delx = np.zeros(len(idx[0]))
  gradz_delx = np.zeros(len(idx[0]))
  
  for ii in range(len(idx[0])):
    xx = idx[0][ii]
    yy = idx[1][ii]
    zz = idx[2][ii]
    pix = eigvecs[xx,yy-1,zz]
    right = eigvecs[xx+1,yy-1,zz]
    right2 = eigvecs[xx+2,yy-1,zz]
    bottom2 = eigvecs[xx,yy-2,zz]
    rightbottom2 = eigvecs[xx+1,yy-2,zz]
    
    if np.dot(pix, right) < 0:
      right = -right
    if np.dot(pix, right2) < 0:
      right2 = -right2

    gradx_delx[ii] = 3*right[0] - 0.5*right2[0] -rightbottom2[0] + bottom2[0] - 2.5*pix[0]
    grady_delx[ii] = 3*right[1] - 0.5*right2[1] -rightbottom2[1] + bottom2[1] - 2.5*pix[1]
    gradz_delx[ii] = 3*right[2] - 0.5*right2[2] -rightbottom2[2] + bottom2[2] - 2.5*pix[2]
  return(gradx_delx, grady_delx, gradz_delx)

def eigv_left_shiftrear_idx_3d(eigvecs, idx):
  # Return the gradient of each component of the eigenvector taken in the x direction, shifted in the z direction
  gradx_delx = np.zeros(len(idx[0]))
  grady_delx = np.zeros(len(idx[0]))
  gradz_delx = np.zeros(len(idx[0]))
  
  for ii in range(len(idx[0])):
    xx = idx[0][ii]
    yy = idx[1][ii]
    zz = idx[2][ii]
    pix = eigvecs[xx,yy,zz+1]
    right = eigvecs[xx+1,yy,zz+1]
    right2 = eigvecs[xx+2,yy,zz+1]
    front2 = eigvecs[xx,yy,zz+2]
    rightfront2 = eigvecs[xx+1,yy,zz+2]
    
    if np.dot(pix, right) < 0:
      right = -right
    if np.dot(pix, right2) < 0:
      right2 = -right2

    gradx_delx[ii] = 3*right[0] - 0.5*right2[0] -rightfront2[0] + front2[0] - 2.5*pix[0]
    grady_delx[ii] = 3*right[1] - 0.5*right2[1] -rightfront2[1] + front2[1] - 2.5*pix[1]
    gradz_delx[ii] = 3*right[2] - 0.5*right2[2] -rightfront2[2] + front2[2] - 2.5*pix[2]
  return(gradx_delx, grady_delx, gradz_delx)

def eigv_left_shiftfront_idx_3d(eigvecs, idx):
  # Return the gradient of each component of the eigenvector taken in the x direction, shifted in the z direction
  gradx_delx = np.zeros(len(idx[0]))
  grady_delx = np.zeros(len(idx[0]))
  gradz_delx = np.zeros(len(idx[0]))
  
  for ii in range(len(idx[0])):
    xx = idx[0][ii]
    yy = idx[1][ii]
    zz = idx[2][ii]
    pix = eigvecs[xx,yy,zz-1]
    right = eigvecs[xx+1,yy,zz-1]
    right2 = eigvecs[xx+2,yy,zz-1]
    rear2 = eigvecs[xx,yy,zz-2]
    rightrear2 = eigvecs[xx+1,yy,zz-2]
    
    if np.dot(pix, right) < 0:
      right = -right
    if np.dot(pix, right2) < 0:
      right2 = -right2

    gradx_delx[ii] = 3*right[0] - 0.5*right2[0] -rightrear2[0] + rear2[0] - 2.5*pix[0]
    grady_delx[ii] = 3*right[1] - 0.5*right2[1] -rightrear2[1] + rear2[1] - 2.5*pix[1]
    gradz_delx[ii] = 3*right[2] - 0.5*right2[2] -rightrear2[2] + rear2[2] - 2.5*pix[2]
  return(gradx_delx, grady_delx, gradz_delx)

def eigv_right_shiftbottom_idx_3d(eigvecs, idx):
  # Return the gradient of each component of the eigenvector taken in the x direction, shifted in the y direction
  gradx_delx = np.zeros(len(idx[0]))
  grady_delx = np.zeros(len(idx[0]))
  gradz_delx = np.zeros(len(idx[0]))
  
  for ii in range(len(idx[0])):
    xx = idx[0][ii]
    yy = idx[1][ii]
    zz = idx[2][ii]
    pix = eigvecs[xx,yy+1,zz]
    left = eigvecs[xx-1,yy+1,zz]
    left2 = eigvecs[xx-2,yy+1,zz]
    top2 = eigvecs[xx,yy+2,zz]
    lefttop2 = eigvecs[xx-1,yy+2,zz]
    
    if np.dot(pix, left) < 0:
      left = -left
    if np.dot(pix, left2) < 0:
      left2 = -left2

    # 2nd order accurate gradx at [i,j,k] = -3a[i-1,j+1,k] + 0.5a[i-2,j+1,k] + a[i-1,j+2,k]
    #                                       -a[i,j+2,k] + 2.5a[i,j+1,k]
    gradx_delx[ii] = -3*left[0] + 0.5*left2[0] + lefttop2[0] - top2[0] + 2.5*pix[0]
    grady_delx[ii] = -3*left[1] + 0.5*left2[1] + lefttop2[1] - top2[1] + 2.5*pix[1]
    gradz_delx[ii] = -3*left[2] + 0.5*left2[2] + lefttop2[2] - top2[2] + 2.5*pix[2]
  return(gradx_delx, grady_delx, gradz_delx)

def eigv_right_shifttop_idx_3d(eigvecs, idx):
  # Return the gradient of each component of the eigenvector taken in the x direction, shifted in the y direction
  gradx_delx = np.zeros(len(idx[0]))
  grady_delx = np.zeros(len(idx[0]))
  gradz_delx = np.zeros(len(idx[0]))
  
  for ii in range(len(idx[0])):
    xx = idx[0][ii]
    yy = idx[1][ii]
    zz = idx[2][ii]
    pix = eigvecs[xx,yy-1,zz]
    left = eigvecs[xx-1,yy-1,zz]
    left2 = eigvecs[xx-2,yy-1,zz]
    bottom2 = eigvecs[xx,yy-2,zz]
    leftbottom2 = eigvecs[xx-1,yy-2,zz]
    
    if np.dot(pix, left) < 0:
      left = -left
    if np.dot(pix, left2) < 0:
      left2 = -left2

    # 2nd order accurate gradx at [i,j,k] = -3a[i-1,j+1,k] + 0.5a[i-2,j+1,k] + a[i-1,j+2,k]
    #                                       -a[i,j+2,k] + 2.5a[i,j+1,k]
    gradx_delx[ii] = -3*left[0] + 0.5*left2[0] + leftbottom2[0] - bottom2[0] + 2.5*pix[0]
    grady_delx[ii] = -3*left[1] + 0.5*left2[1] + leftbottom2[1] - bottom2[1] + 2.5*pix[1]
    gradz_delx[ii] = -3*left[2] + 0.5*left2[2] + leftbottom2[2] - bottom2[2] + 2.5*pix[2]
  return(gradx_delx, grady_delx, gradz_delx)

def eigv_right_shiftrear_idx_3d(eigvecs, idx):
  # Return the gradient of each component of the eigenvector taken in the x direction, shifted in the z direction
  gradx_delx = np.zeros(len(idx[0]))
  grady_delx = np.zeros(len(idx[0]))
  gradz_delx = np.zeros(len(idx[0]))
  
  for ii in range(len(idx[0])):
    xx = idx[0][ii]
    yy = idx[1][ii]
    zz = idx[2][ii]
    pix = eigvecs[xx,yy,zz+1]
    left = eigvecs[xx-1,yy,zz+1]
    left2 = eigvecs[xx-2,yy,zz+1]
    front2 = eigvecs[xx,yy,zz+2]
    leftfront2 = eigvecs[xx-1,yy,zz+2]
    
    if np.dot(pix, left) < 0:
      left = -left
    if np.dot(pix, left2) < 0:
      left2 = -left2

    # 2nd order accurate gradx at [i,j,k] = -3a[i-1,j+1,k] + 0.5a[i-2,j+1,k] + a[i-1,j+2,k]
    #                                       -a[i,j+2,k] + 2.5a[i,j+1,k]
    gradx_delx[ii] = -3*left[0] + 0.5*left2[0] + leftfront2[0] - front2[0] + 2.5*pix[0]
    grady_delx[ii] = -3*left[1] + 0.5*left2[1] + leftfront2[1] - front2[1] + 2.5*pix[1]
    gradz_delx[ii] = -3*left[2] + 0.5*left2[2] + leftfront2[2] - front2[2] + 2.5*pix[2]
  return(gradx_delx, grady_delx, gradz_delx)

def eigv_right_shiftfront_idx_3d(eigvecs, idx):
  # Return the gradient of each component of the eigenvector taken in the x direction, shifted in the z direction
  gradx_delx = np.zeros(len(idx[0]))
  grady_delx = np.zeros(len(idx[0]))
  gradz_delx = np.zeros(len(idx[0]))
  
  for ii in range(len(idx[0])):
    xx = idx[0][ii]
    yy = idx[1][ii]
    zz = idx[2][ii]
    pix = eigvecs[xx,yy,zz-1]
    left = eigvecs[xx-1,yy,zz-1]
    left2 = eigvecs[xx-2,yy,zz-1]
    rear2 = eigvecs[xx,yy,zz-2]
    leftrear2 = eigvecs[xx-1,yy,zz-2]
    
    if np.dot(pix, left) < 0:
      left = -left
    if np.dot(pix, left2) < 0:
      left2 = -left2

    # 2nd order accurate gradx at [i,j,k] = -3a[i-1,j+1,k] + 0.5a[i-2,j+1,k] + a[i-1,j+2,k]
    #                                       -a[i,j+2,k] + 2.5a[i,j+1,k]
    gradx_delx[ii] = -3*left[0] + 0.5*left2[0] + leftrear2[0] - rear2[0] + 2.5*pix[0]
    grady_delx[ii] = -3*left[1] + 0.5*left2[1] + leftrear2[1] - rear2[1] + 2.5*pix[1]
    gradz_delx[ii] = -3*left[2] + 0.5*left2[2] + leftrear2[2] - rear2[2] + 2.5*pix[2]
  return(gradx_delx, grady_delx, gradz_delx)

def eigv_grady_shiftleft_idx_3d(eigvecs, idx):
  # Return the gradient of each component of the eigenvector taken in the y direction shifted in the x direction
  gradx_dely = np.zeros(len(idx[0]))
  grady_dely = np.zeros(len(idx[0]))
  gradz_dely = np.zeros(len(idx[0]))
  
  for ii in range(len(idx[0])):
    xx = idx[0][ii]
    yy = idx[1][ii]
    zz = idx[2][ii]
    pix = eigvecs[xx+1,yy,zz]
    bottom = eigvecs[xx+1,yy-1,zz]
    top = eigvecs[xx+1,yy+1,zz]
    bottom2 = eigvecs[xx+2,yy-1,zz]
    top2 = eigvecs[xx+2,yy+1,zz]

    if np.dot(top, bottom) < 0:
      if np.dot(top, pix) < 0:
        top = -top
      else:
        bottom = -bottom

 # 2nd order accurate grady at [i,j,k] = 0.5a[i+1,j+1,k]-0.5a[i+1,j-1,k]+0.25a[i+2,j+1,k]-0.25a[i+2,j-1,k]
    gradx_dely[ii] = 0.5 * top[0] - 0.5 * bottom[0] + 0.25 * top2[0] -0.25 * bottom2[0]
    grady_dely[ii] = 0.5 * top[1] - 0.5 * bottom[1] + 0.25 * top2[1] -0.25 * bottom2[1]
    gradz_dely[ii] = 0.5 * top[2] - 0.5 * bottom[2] + 0.25 * top2[2] -0.25 * bottom2[2]
  return(gradx_dely, grady_dely, gradz_dely)

def eigv_grady_shiftright_idx_3d(eigvecs, idx):
  # Return the gradient of each component of the eigenvector taken in the y direction shifted in the x direction
  gradx_dely = np.zeros(len(idx[0]))
  grady_dely = np.zeros(len(idx[0]))
  gradz_dely = np.zeros(len(idx[0]))
  
  for ii in range(len(idx[0])):
    xx = idx[0][ii]
    yy = idx[1][ii]
    zz = idx[2][ii]
    pix = eigvecs[xx-1,yy,zz]
    bottom = eigvecs[xx-1,yy-1,zz]
    top = eigvecs[xx-1,yy+1,zz]
    bottom2 = eigvecs[xx-2,yy-1,zz]
    top2 = eigvecs[xx-2,yy+1,zz]

    if np.dot(top, bottom) < 0:
      if np.dot(top, pix) < 0:
        top = -top
      else:
        bottom = -bottom
        
    # 2nd order accurate gradx at [i,j,k] = a[i+1,j-1,k]-a[i-1,j-1,k]-0.5a[i+1,j-2,k]+0.5a[i-1,j-2,k]
    gradx_dely[ii] = top[0] - bottom[0] - 0.5 * top2[0] + 0.5 * bottom2[0]
    grady_dely[ii] = top[1] - bottom[1] - 0.5 * top2[1] + 0.5 * bottom2[1]
    gradz_dely[ii] = top[2] - bottom[2] - 0.5 * top2[2] + 0.5 * bottom2[2]
  return(gradx_dely, grady_dely, gradz_dely)

def eigv_grady_shiftrear_idx_3d(eigvecs, idx):
  # Return the gradient of each component of the eigenvector taken in the y direction shifted in the z direction
  gradx_dely = np.zeros(len(idx[0]))
  grady_dely = np.zeros(len(idx[0]))
  gradz_dely = np.zeros(len(idx[0]))
  
  for ii in range(len(idx[0])):
    xx = idx[0][ii]
    yy = idx[1][ii]
    zz = idx[2][ii]
    pix = eigvecs[xx,yy,zz+1]
    bottom = eigvecs[xx,yy-1,zz+1]
    top = eigvecs[xx,yy+1,zz+1]
    bottom2 = eigvecs[xx,yy-1,zz+2]
    top2 = eigvecs[xx,yy+1,zz+2]

    if np.dot(top, bottom) < 0:
      if np.dot(top, pix) < 0:
        top = -top
      else:
        bottom = -bottom

    gradx_dely[ii] = 0.5 * top[0] - 0.5 * bottom[0] + 0.25 * top2[0] -0.25 * bottom2[0]
    grady_dely[ii] = 0.5 * top[1] - 0.5 * bottom[1] + 0.25 * top2[1] -0.25 * bottom2[1]
    gradz_dely[ii] = 0.5 * top[2] - 0.5 * bottom[2] + 0.25 * top2[2] -0.25 * bottom2[2]
  return(gradx_dely, grady_dely, gradz_dely)

def eigv_grady_shiftfront_idx_3d(eigvecs, idx):
  # Return the gradient of each component of the eigenvector taken in the y direction shifted in the z direction
  gradx_dely = np.zeros(len(idx[0]))
  grady_dely = np.zeros(len(idx[0]))
  gradz_dely = np.zeros(len(idx[0]))
  
  for ii in range(len(idx[0])):
    xx = idx[0][ii]
    yy = idx[1][ii]
    zz = idx[2][ii]
    pix = eigvecs[xx,yy,zz-1]
    bottom = eigvecs[xx,yy-1,zz-1]
    top = eigvecs[xx,yy+1,zz-1]
    bottom2 = eigvecs[xx,yy-1,zz-2]
    top2 = eigvecs[xx,yy+1,zz-2]

    if np.dot(top, bottom) < 0:
      if np.dot(top, pix) < 0:
        top = -top
      else:
        bottom = -bottom

    # 2nd order accurate gradx at [i,j,k] = a[i+1,j-1,k]-a[i-1,j-1,k]-0.5a[i+1,j-2,k]+0.5a[i-1,j-2,k]
    gradx_dely[ii] = top[0] - bottom[0] - 0.5 * top2[0] + 0.5 * bottom2[0]
    grady_dely[ii] = top[1] - bottom[1] - 0.5 * top2[1] + 0.5 * bottom2[1]
    gradz_dely[ii] = top[2] - bottom[2] - 0.5 * top2[2] + 0.5 * bottom2[2]
  return(gradx_dely, grady_dely, gradz_dely)

def eigv_bottom_shiftleft_idx_3d(eigvecs, idx):
  # Return the gradient of each component of the eigenvector taken in the y direction shifted in the x direction
  gradx_dely = np.zeros(len(idx[0]))
  grady_dely = np.zeros(len(idx[0]))
  gradz_dely = np.zeros(len(idx[0]))
  
  for ii in range(len(idx[0])):
    xx = idx[0][ii]
    yy = idx[1][ii]
    zz = idx[2][ii]
    pix = eigvecs[xx+1,yy,zz]
    top= eigvecs[xx+1,yy+1,zz]
    top2 = eigvecs[xx+1,yy+2,zz]
    right2 = eigvecs[xx+2,yy,zz]
    topright2 = eigvecs[xx+2,yy+1,zz]

    if np.dot(pix, top) < 0:
      top = -top
    if np.dot(pix, top2) < 0:
      top2 = -top2

    gradx_dely[ii] = 3*top[0] - 0.5*top2[0] -topright2[0] + right2[0] - 2.5*pix[0]
    grady_dely[ii] = 3*top[1] - 0.5*top2[1] -topright2[1] + right2[1] - 2.5*pix[1]
    gradz_dely[ii] = 3*top[2] - 0.5*top2[2] -topright2[2] + right2[2] - 2.5*pix[2]
  return(gradx_dely, grady_dely, gradz_dely)

def eigv_bottom_shiftright_idx_3d(eigvecs, idx):
  # Return the gradient of each component of the eigenvector taken in the y direction shifted in the x direction
  gradx_dely = np.zeros(len(idx[0]))
  grady_dely = np.zeros(len(idx[0]))
  gradz_dely = np.zeros(len(idx[0]))
  
  for ii in range(len(idx[0])):
    xx = idx[0][ii]
    yy = idx[1][ii]
    zz = idx[2][ii]
    pix = eigvecs[xx-1,yy,zz]
    top= eigvecs[xx-1,yy+1,zz]
    top2 = eigvecs[xx-1,yy+2,zz]
    left2 = eigvecs[xx-2,yy,zz]
    topleft2 = eigvecs[xx-2,yy+1,zz]

    if np.dot(pix, top) < 0:
      top = -top
    if np.dot(pix, top2) < 0:
      top2 = -top2

    gradx_dely[ii] = 3*top[0] - 0.5*top2[0] -topleft2[0] + left2[0] - 2.5*pix[0]
    grady_dely[ii] = 3*top[1] - 0.5*top2[1] -topleft2[1] + left2[1] - 2.5*pix[1]
    gradz_dely[ii] = 3*top[2] - 0.5*top2[2] -topleft2[2] + left2[2] - 2.5*pix[2]
  return(gradx_dely, grady_dely, gradz_dely)

def eigv_bottom_shiftrear_idx_3d(eigvecs, idx):
  # Return the gradient of each component of the eigenvector taken in the y direction shifted in the x direction
  gradx_dely = np.zeros(len(idx[0]))
  grady_dely = np.zeros(len(idx[0]))
  gradz_dely = np.zeros(len(idx[0]))
  
  for ii in range(len(idx[0])):
    xx = idx[0][ii]
    yy = idx[1][ii]
    zz = idx[2][ii]
    pix = eigvecs[xx,yy,zz+1]
    top= eigvecs[xx,yy+1,zz+1]
    top2 = eigvecs[xx,yy+2,zz+1]
    front2 = eigvecs[xx,yy,zz+2]
    topfront2 = eigvecs[xx,yy+1,zz+2]

    if np.dot(pix, top) < 0:
      top = -top
    if np.dot(pix, top2) < 0:
      top2 = -top2

    gradx_dely[ii] = 3*top[0] - 0.5*top2[0] -topfront2[0] + front2[0] - 2.5*pix[0]
    grady_dely[ii] = 3*top[1] - 0.5*top2[1] -topfront2[1] + front2[1] - 2.5*pix[1]
    gradz_dely[ii] = 3*top[2] - 0.5*top2[2] -topfront2[2] + front2[2] - 2.5*pix[2]
  return(gradx_dely, grady_dely, gradz_dely)

def eigv_bottom_shiftfront_idx_3d(eigvecs, idx):
  # Return the gradient of each component of the eigenvector taken in the y direction shifted in the x direction
  gradx_dely = np.zeros(len(idx[0]))
  grady_dely = np.zeros(len(idx[0]))
  gradz_dely = np.zeros(len(idx[0]))
  
  for ii in range(len(idx[0])):
    xx = idx[0][ii]
    yy = idx[1][ii]
    zz = idx[2][ii]
    pix = eigvecs[xx,yy,zz-1]
    top= eigvecs[xx,yy+1,zz-1]
    top2 = eigvecs[xx,yy+2,zz-1]
    rear2 = eigvecs[xx,yy,zz-2]
    toprear2 = eigvecs[xx,yy+1,zz-2]

    if np.dot(pix, top) < 0:
      top = -top
    if np.dot(pix, top2) < 0:
      top2 = -top2

    gradx_dely[ii] = 3*top[0] - 0.5*top2[0] -toprear2[0] + rear2[0] - 2.5*pix[0]
    grady_dely[ii] = 3*top[1] - 0.5*top2[1] -toprear2[1] + rear2[1] - 2.5*pix[1]
    gradz_dely[ii] = 3*top[2] - 0.5*top2[2] -toprear2[2] + rear2[2] - 2.5*pix[2]
  return(gradx_dely, grady_dely, gradz_dely)

def eigv_top_shiftleft_idx_3d(eigvecs, idx):
  # Return the gradient of each component of the eigenvector taken in the y direction shifted in the x direction
  gradx_dely = np.zeros(len(idx[0]))
  grady_dely = np.zeros(len(idx[0]))
  gradz_dely = np.zeros(len(idx[0]))
  
  for ii in range(len(idx[0])):
    xx = idx[0][ii]
    yy = idx[1][ii]
    zz = idx[2][ii]
    pix = eigvecs[xx+1,yy,zz]
    bottom= eigvecs[xx+1,yy-1,zz]
    bottom2 = eigvecs[xx+1,yy-2,zz]
    right2 = eigvecs[xx+2,yy,zz]
    bottomright2 = eigvecs[xx+2,yy-1,zz]

    if np.dot(pix, bottom) < 0:
      bottom = -bottom
    if np.dot(pix, bottom2) < 0:
      bottom2 = -bottom2

    gradx_dely[ii] = -3*bottom[0] + 0.5*bottom2[0] + bottomright2[0] - right2[0] + 2.5*pix[0]
    grady_dely[ii] = -3*bottom[1] + 0.5*bottom2[1] + bottomright2[1] - right2[1] + 2.5*pix[1]
    gradz_dely[ii] = -3*bottom[2] + 0.5*bottom2[2] + bottomright2[2] - right2[2] + 2.5*pix[2]
  return(gradx_dely, grady_dely, gradz_dely)

def eigv_top_shiftright_idx_3d(eigvecs, idx):
  # Return the gradient of each component of the eigenvector taken in the y direction shifted in the x direction
  gradx_dely = np.zeros(len(idx[0]))
  grady_dely = np.zeros(len(idx[0]))
  gradz_dely = np.zeros(len(idx[0]))
  
  for ii in range(len(idx[0])):
    xx = idx[0][ii]
    yy = idx[1][ii]
    zz = idx[2][ii]
    pix = eigvecs[xx-1,yy,zz]
    bottom= eigvecs[xx-1,yy-1,zz]
    bottom2 = eigvecs[xx-1,yy-2,zz]
    left2 = eigvecs[xx-2,yy,zz]
    bottomleft2 = eigvecs[xx-2,yy-1,zz]

    if np.dot(pix, bottom) < 0:
      bottom = -bottom
    if np.dot(pix, bottom2) < 0:
      bottom2 = -bottom2

    gradx_dely[ii] = -3*bottom[0] + 0.5*bottom2[0] + bottomleft2[0] - left2[0] + 2.5*pix[0]
    grady_dely[ii] = -3*bottom[1] + 0.5*bottom2[1] + bottomleft2[1] - left2[1] + 2.5*pix[1]
    gradz_dely[ii] = -3*bottom[2] + 0.5*bottom2[2] + bottomleft2[2] - left2[2] + 2.5*pix[2]
  return(gradx_dely, grady_dely, gradz_dely)

def eigv_top_shiftrear_idx_3d(eigvecs, idx):
  # Return the gradient of each component of the eigenvector taken in the y direction shifted in the x direction
  gradx_dely = np.zeros(len(idx[0]))
  grady_dely = np.zeros(len(idx[0]))
  gradz_dely = np.zeros(len(idx[0]))
  
  for ii in range(len(idx[0])):
    xx = idx[0][ii]
    yy = idx[1][ii]
    zz = idx[2][ii]
    pix = eigvecs[xx,yy,zz+1]
    bottom= eigvecs[xx,yy-1,zz+1]
    bottom2 = eigvecs[xx,yy-2,zz+1]
    front2 = eigvecs[xx,yy,zz+2]
    bottomfront2 = eigvecs[xx,yy-1,zz+2]

    if np.dot(pix, bottom) < 0:
      bottom = -bottom
    if np.dot(pix, bottom2) < 0:
      bottom2 = -bottom2

    gradx_dely[ii] = -3*bottom[0] + 0.5*bottom2[0] + bottomfront2[0] - front2[0] + 2.5*pix[0]
    grady_dely[ii] = -3*bottom[1] + 0.5*bottom2[1] + bottomfront2[1] - front2[1] + 2.5*pix[1]
    gradz_dely[ii] = -3*bottom[2] + 0.5*bottom2[2] + bottomfront2[2] - front2[2] + 2.5*pix[2]
  return(gradx_dely, grady_dely, gradz_dely)

def eigv_top_shiftfront_idx_3d(eigvecs, idx):
  # Return the gradient of each component of the eigenvector taken in the y direction shifted in the x direction
  gradx_dely = np.zeros(len(idx[0]))
  grady_dely = np.zeros(len(idx[0]))
  gradz_dely = np.zeros(len(idx[0]))
  
  for ii in range(len(idx[0])):
    xx = idx[0][ii]
    yy = idx[1][ii]
    zz = idx[2][ii]
    pix = eigvecs[xx,yy,zz-1]
    bottom= eigvecs[xx,yy-1,zz-1]
    bottom2 = eigvecs[xx,yy-2,zz-1]
    rear2 = eigvecs[xx,yy,zz-2]
    bottomrear2 = eigvecs[xx,yy-1,zz-2]

    if np.dot(pix, bottom) < 0:
      bottom = -bottom
    if np.dot(pix, bottom2) < 0:
      bottom2 = -bottom2

    gradx_dely[ii] = -3*bottom[0] + 0.5*bottom2[0] + bottomrear2[0] - rear2[0] + 2.5*pix[0]
    grady_dely[ii] = -3*bottom[1] + 0.5*bottom2[1] + bottomrear2[1] - rear2[1] + 2.5*pix[1]
    gradz_dely[ii] = -3*bottom[2] + 0.5*bottom2[2] + bottomrear2[2] - rear2[2] + 2.5*pix[2]
  return(gradx_dely, grady_dely, gradz_dely)

def eigv_gradz_shiftleft_idx_3d(eigvecs, idx):
  # Return the gradient of each component of the eigenvector taken in the z direction shifted in the x direction
  gradx_delz = np.zeros(len(idx[0]))
  grady_delz = np.zeros(len(idx[0]))
  gradz_delz = np.zeros(len(idx[0]))
  
  for ii in range(len(idx[0])):
    xx = idx[0][ii]
    yy = idx[1][ii]
    zz = idx[2][ii]
    pix = eigvecs[xx+1,yy,zz]
    rear = eigvecs[xx+1,yy,zz-1]
    front = eigvecs[xx+1,yy,zz+1]
    rear2 = eigvecs[xx+2,yy,zz-1]
    front2 = eigvecs[xx+2,yy,zz+1]

    if np.dot(front, rear) < 0:
      if np.dot(front, pix) < 0:
        front = -front
      else:
        rear = -rear

    gradx_delz[ii] = 0.5 * front[0] - 0.5 * rear[0] + 0.25 * front2[0] -0.25 * rear2[0]
    grady_delz[ii] = 0.5 * front[1] - 0.5 * rear[1] + 0.25 * front2[1] -0.25 * rear2[1]
    gradz_delz[ii] = 0.5 * front[2] - 0.5 * rear[2] + 0.25 * front2[2] -0.25 * rear2[2]
  return(gradx_delz, grady_delz, gradz_delz)

def eigv_gradz_shiftright_idx_3d(eigvecs, idx):
  # Return the gradient of each component of the eigenvector taken in the z direction shifted in the x direction
  gradx_delz = np.zeros(len(idx[0]))
  grady_delz = np.zeros(len(idx[0]))
  gradz_delz = np.zeros(len(idx[0]))
  
  for ii in range(len(idx[0])):
    xx = idx[0][ii]
    yy = idx[1][ii]
    zz = idx[2][ii]
    pix = eigvecs[xx-1,yy,zz]
    rear = eigvecs[xx-1,yy,zz-1]
    front = eigvecs[xx-1,yy,zz+1]
    rear2 = eigvecs[xx-2,yy,zz-1]
    front2 = eigvecs[xx-2,yy,zz+1]

    if np.dot(front, rear) < 0:
      if np.dot(front, pix) < 0:
        front = -front
      else:
        rear = -rear

    # 2nd order accurate gradx at [i,j,k] = a[i+1,j-1,k]-a[i-1,j-1,k]-0.5a[i+1,j-2,k]+0.5a[i-1,j-2,k]
    gradx_delz[ii] = front[0] - rear[0] - 0.5 * front2[0] + 0.5 * rear2[0]
    grady_delz[ii] = front[1] - rear[1] - 0.5 * front2[1] + 0.5 * rear2[1]
    gradz_delz[ii] = front[2] - rear[2] - 0.5 * front2[2] + 0.5 * rear2[2]
  return(gradx_delz, grady_delz, gradz_delz)

def eigv_gradz_shiftbottom_idx_3d(eigvecs, idx):
  # Return the gradient of each component of the eigenvector taken in the z direction shifted in the y direction
  gradx_delz = np.zeros(len(idx[0]))
  grady_delz = np.zeros(len(idx[0]))
  gradz_delz = np.zeros(len(idx[0]))
  
  for ii in range(len(idx[0])):
    xx = idx[0][ii]
    yy = idx[1][ii]
    zz = idx[2][ii]
    pix = eigvecs[xx,yy+1,zz]
    rear = eigvecs[xx,yy+1,zz-1]
    front = eigvecs[xx,yy+1,zz+1]
    rear2 = eigvecs[xx,yy+2,zz-1]
    front2 = eigvecs[xx,yy+2,zz+1]

    if np.dot(front, rear) < 0:
      if np.dot(front, pix) < 0:
        front = -front
      else:
        rear = -rear

    gradx_delz[ii] = 0.5 * front[0] - 0.5 * rear[0] + 0.25 * front2[0] -0.25 * rear2[0]
    grady_delz[ii] = 0.5 * front[1] - 0.5 * rear[1] + 0.25 * front2[1] -0.25 * rear2[1]
    gradz_delz[ii] = 0.5 * front[2] - 0.5 * rear[2] + 0.25 * front2[2] -0.25 * rear2[2]
  return(gradx_delz, grady_delz, gradz_delz)

def eigv_gradz_shifttop_idx_3d(eigvecs, idx):
  # Return the gradient of each component of the eigenvector taken in the z direction shifted in the y direction
  gradx_delz = np.zeros(len(idx[0]))
  grady_delz = np.zeros(len(idx[0]))
  gradz_delz = np.zeros(len(idx[0]))
  
  for ii in range(len(idx[0])):
    xx = idx[0][ii]
    yy = idx[1][ii]
    zz = idx[2][ii]
    pix = eigvecs[xx,yy-1,zz]
    rear = eigvecs[xx,yy-1,zz-1]
    front = eigvecs[xx,yy-1,zz+1]
    rear2 = eigvecs[xx,yy-2,zz-1]
    front2 = eigvecs[xx,yy-2,zz+1]

    if np.dot(front, rear) < 0:
      if np.dot(front, pix) < 0:
        front = -front
      else:
        rear = -rear

    # 2nd order accurate gradx at [i,j,k] = a[i+1,j-1,k]-a[i-1,j-1,k]-0.5a[i+1,j-2,k]+0.5a[i-1,j-2,k]
    gradx_delz[ii] = front[0] - rear[0] - 0.5 * front2[0] + 0.5 * rear2[0]
    grady_delz[ii] = front[1] - rear[1] - 0.5 * front2[1] + 0.5 * rear2[1]
    gradz_delz[ii] = front[2] - rear[2] - 0.5 * front2[2] + 0.5 * rear2[2]
  return(gradx_delz, grady_delz, gradz_delz)

def eigv_rear_shiftleft_idx_3d(eigvecs, idx):
  # Return the gradient of each component of the eigenvector taken in the z direction shifted in the x direction
  gradx_delz = np.zeros(len(idx[0]))
  grady_delz = np.zeros(len(idx[0]))
  gradz_delz = np.zeros(len(idx[0]))
  
  for ii in range(len(idx[0])):
    xx = idx[0][ii]
    yy = idx[1][ii]
    zz = idx[2][ii]
    pix = eigvecs[xx+1,yy,zz]
    front= eigvecs[xx+1,yy,zz+1]
    front2 = eigvecs[xx+1,yy,zz+2]
    right2 = eigvecs[xx+2,yy,zz]
    frontright2 = eigvecs[xx+2,yy,zz+1]

    if np.dot(pix, front) < 0:
      front = -front
    if np.dot(pix, front2) < 0:
      front2 = -front2

    gradx_delz[ii] = 3*front[0] - 0.5*front2[0] -frontright2[0] + right2[0] - 2.5*pix[0]
    grady_delz[ii] = 3*front[1] - 0.5*front2[1] -frontright2[1] + right2[1] - 2.5*pix[1]
    gradz_delz[ii] = 3*front[2] - 0.5*front2[2] -frontright2[2] + right2[2] - 2.5*pix[2]
  return(gradx_delz, grady_delz, gradz_delz)

def eigv_rear_shiftright_idx_3d(eigvecs, idx):
  # Return the gradient of each component of the eigenvector taken in the z direction shifted in the x direction
  gradx_delz = np.zeros(len(idx[0]))
  grady_delz = np.zeros(len(idx[0]))
  gradz_delz = np.zeros(len(idx[0]))
  
  for ii in range(len(idx[0])):
    xx = idx[0][ii]
    yy = idx[1][ii]
    zz = idx[2][ii]
    pix = eigvecs[xx-1,yy,zz]
    front= eigvecs[xx-1,yy,zz+1]
    front2 = eigvecs[xx-1,yy,zz+2]
    left2 = eigvecs[xx-2,yy,zz]
    frontleft2 = eigvecs[xx-2,yy,zz+1]

    if np.dot(pix, front) < 0:
      front = -front
    if np.dot(pix, front2) < 0:
      front2 = -front2

    gradx_delz[ii] = 3*front[0] - 0.5*front2[0] -frontleft2[0] + left2[0] - 2.5*pix[0]
    grady_delz[ii] = 3*front[1] - 0.5*front2[1] -frontleft2[1] + left2[1] - 2.5*pix[1]
    gradz_delz[ii] = 3*front[2] - 0.5*front2[2] -frontleft2[2] + left2[2] - 2.5*pix[2]
  return(gradx_delz, grady_delz, gradz_delz)

def eigv_rear_shiftbottom_idx_3d(eigvecs, idx):
  # Return the gradient of each component of the eigenvector taken in the z direction shifted in the y direction
  gradx_delz = np.zeros(len(idx[0]))
  grady_delz = np.zeros(len(idx[0]))
  gradz_delz = np.zeros(len(idx[0]))
  
  for ii in range(len(idx[0])):
    xx = idx[0][ii]
    yy = idx[1][ii]
    zz = idx[2][ii]
    pix = eigvecs[xx,yy+1,zz]
    front= eigvecs[xx,yy+1,zz+1]
    front2 = eigvecs[xx,yy+1,zz+2]
    top2 = eigvecs[xx,yy+2,zz]
    fronttop2 = eigvecs[xx,yy+2,zz+1]

    if np.dot(pix, front) < 0:
      front = -front
    if np.dot(pix, front2) < 0:
      front2 = -front2

    gradx_delz[ii] = 3*front[0] - 0.5*front2[0] -fronttop2[0] + top2[0] - 2.5*pix[0]
    grady_delz[ii] = 3*front[1] - 0.5*front2[1] -fronttop2[1] + top2[1] - 2.5*pix[1]
    gradz_delz[ii] = 3*front[2] - 0.5*front2[2] -fronttop2[2] + top2[2] - 2.5*pix[2]
  return(gradx_delz, grady_delz, gradz_delz)

def eigv_rear_shifttop_idx_3d(eigvecs, idx):
  # Return the gradient of each component of the eigenvector taken in the z direction shifted in the y direction
  gradx_delz = np.zeros(len(idx[0]))
  grady_delz = np.zeros(len(idx[0]))
  gradz_delz = np.zeros(len(idx[0]))
  
  for ii in range(len(idx[0])):
    xx = idx[0][ii]
    yy = idx[1][ii]
    zz = idx[2][ii]
    pix = eigvecs[xx,yy-1,zz]
    front= eigvecs[xx,yy-1,zz+1]
    front2 = eigvecs[xx,yy-1,zz+2]
    bottom2 = eigvecs[xx,yy-2,zz]
    frontbottom2 = eigvecs[xx,yy-2,zz+1]

    if np.dot(pix, front) < 0:
      front = -front
    if np.dot(pix, front2) < 0:
      front2 = -front2

    gradx_delz[ii] = 3*front[0] - 0.5*front2[0] -frontbottom2[0] + bottom2[0] - 2.5*pix[0]
    grady_delz[ii] = 3*front[1] - 0.5*front2[1] -frontbottom2[1] + bottom2[1] - 2.5*pix[1]
    gradz_delz[ii] = 3*front[2] - 0.5*front2[2] -frontbottom2[2] + bottom2[2] - 2.5*pix[2]
  return(gradx_delz, grady_delz, gradz_delz)

def eigv_front_shiftleft_idx_3d(eigvecs, idx):
  # Return the gradient of each component of the eigenvector taken in the z direction shifted in the x direction
  gradx_delz = np.zeros(len(idx[0]))
  grady_delz = np.zeros(len(idx[0]))
  gradz_delz = np.zeros(len(idx[0]))
  
  for ii in range(len(idx[0])):
    xx = idx[0][ii]
    yy = idx[1][ii]
    zz = idx[2][ii]
    pix = eigvecs[xx+1,yy,zz]
    rear= eigvecs[xx+1,yy,zz-1]
    rear2 = eigvecs[xx+1,yy,zz-2]
    right2 = eigvecs[xx+2,yy,zz]
    rearright2 = eigvecs[xx+2,yy,zz-1]

    if np.dot(pix, rear) < 0:
      rear = -rear
    if np.dot(pix, rear2) < 0:
      rear2 = -rear2

    gradx_delz[ii] = -3*rear[0] + 0.5*rear2[0] + rearright2[0] - right2[0] + 2.5*pix[0]
    grady_delz[ii] = -3*rear[1] + 0.5*rear2[1] + rearright2[1] - right2[1] + 2.5*pix[1]
    gradz_delz[ii] = -3*rear[2] + 0.5*rear2[2] + rearright2[2] - right2[2] + 2.5*pix[2]
  return(gradx_delz, grady_delz, gradz_delz)

def eigv_front_shiftright_idx_3d(eigvecs, idx):
  # Return the gradient of each component of the eigenvector taken in the z direction shifted in the x direction
  gradx_delz = np.zeros(len(idx[0]))
  grady_delz = np.zeros(len(idx[0]))
  gradz_delz = np.zeros(len(idx[0]))
  
  for ii in range(len(idx[0])):
    xx = idx[0][ii]
    yy = idx[1][ii]
    zz = idx[2][ii]
    pix = eigvecs[xx-1,yy,zz]
    rear= eigvecs[xx-1,yy,zz-1]
    rear2 = eigvecs[xx-1,yy,zz-2]
    left2 = eigvecs[xx-2,yy,zz]
    rearleft2 = eigvecs[xx-2,yy,zz-1]

    if np.dot(pix, rear) < 0:
      rear = -rear
    if np.dot(pix, rear2) < 0:
      rear2 = -rear2

    gradx_delz[ii] = -3*rear[0] + 0.5*rear2[0] + rearleft2[0] - left2[0] + 2.5*pix[0]
    grady_delz[ii] = -3*rear[1] + 0.5*rear2[1] + rearleft2[1] - left2[1] + 2.5*pix[1]
    gradz_delz[ii] = -3*rear[2] + 0.5*rear2[2] + rearleft2[2] - left2[2] + 2.5*pix[2]
  return(gradx_delz, grady_delz, gradz_delz)

def eigv_front_shiftbottom_idx_3d(eigvecs, idx):
  # Return the gradient of each component of the eigenvector taken in the z direction shifted in the y direction
  gradx_delz = np.zeros(len(idx[0]))
  grady_delz = np.zeros(len(idx[0]))
  gradz_delz = np.zeros(len(idx[0]))
  
  for ii in range(len(idx[0])):
    xx = idx[0][ii]
    yy = idx[1][ii]
    zz = idx[2][ii]
    pix = eigvecs[xx,yy+1,zz]
    rear= eigvecs[xx,yy+1,zz-1]
    rear2 = eigvecs[xx,yy+1,zz-2]
    top2 = eigvecs[xx,yy+2,zz]
    reartop2 = eigvecs[xx,yy+2,zz-1]

    if np.dot(pix, rear) < 0:
      rear = -rear
    if np.dot(pix, rear2) < 0:
      rear2 = -rear2

    gradx_delz[ii] = -3*rear[0] + 0.5*rear2[0] + reartop2[0] - top2[0] + 2.5*pix[0]
    grady_delz[ii] = -3*rear[1] + 0.5*rear2[1] + reartop2[1] - top2[1] + 2.5*pix[1]
    gradz_delz[ii] = -3*rear[2] + 0.5*rear2[2] + reartop2[2] - top2[2] + 2.5*pix[2]
  return(gradx_delz, grady_delz, gradz_delz)

def eigv_front_shifttop_idx_3d(eigvecs, idx):
  # Return the gradient of each component of the eigenvector taken in the z direction shifted in the y direction
  gradx_delz = np.zeros(len(idx[0]))
  grady_delz = np.zeros(len(idx[0]))
  gradz_delz = np.zeros(len(idx[0]))
  
  for ii in range(len(idx[0])):
    xx = idx[0][ii]
    yy = idx[1][ii]
    zz = idx[2][ii]
    pix = eigvecs[xx,yy-1,zz]
    rear= eigvecs[xx,yy-1,zz-1]
    rear2 = eigvecs[xx,yy-1,zz-2]
    top2 = eigvecs[xx,yy-2,zz]
    reartop2 = eigvecs[xx,yy-2,zz-1]

    if np.dot(pix, rear) < 0:
      rear = -rear
    if np.dot(pix, rear2) < 0:
      rear2 = -rear2

    gradx_delz[ii] = -3*rear[0] + 0.5*rear2[0] + reartop2[0] - top2[0] + 2.5*pix[0]
    grady_delz[ii] = -3*rear[1] + 0.5*rear2[1] + reartop2[1] - top2[1] + 2.5*pix[1]
    gradz_delz[ii] = -3*rear[2] + 0.5*rear2[2] + reartop2[2] - top2[2] + 2.5*pix[2]
  return(gradx_delz, grady_delz, gradz_delz)

def eigv_left_idx_3d(eigvecs, idx):
  # Return the gradient of each component of the eigenvector taken in the x direction
  # using only terms to the right
  # called left, because used for case when left direction (i-1) is out of bounds
  leftx_delx = np.zeros(len(idx[0]))
  lefty_delx = np.zeros(len(idx[0]))
  leftz_delx = np.zeros(len(idx[0]))
  
  for ii in range(len(idx[0])):
    xx = idx[0][ii]
    yy = idx[1][ii]
    zz = idx[2][ii]
    pix = eigvecs[xx,yy,zz]
    right1 = eigvecs[xx+1,yy,zz]
    right2 = eigvecs[xx+2,yy,zz]

    if np.dot(pix, right1) < 0:
      right1 = -right1
    if np.dot(pix, right2) < 0:
      right2 = -right2
    
    #left_x = -0.5 * a[idx[0]+2, idx[1]] + 2.0 * a[idx[0]+1, idx[1]] - 1.5 * a[idx[0], idx[1]]
    leftx_delx[ii] = -0.5 * right2[0] + 2.0 * right1[0] - 1.5 * pix[0]
    lefty_delx[ii] = -0.5 * right2[1] + 2.0 * right1[1] - 1.5 * pix[1]
    leftz_delx[ii] = -0.5 * right2[2] + 2.0 * right1[2] - 1.5 * pix[2]
  return(leftx_delx, lefty_delx, leftz_delx)

def eigv_right_idx_3d(eigvecs, idx):
  # Return the gradient of each component of the eigenvector taken in the x direction
  # using only terms to the left
  # called right, because used for case when right direction (i+1) is out of bounds
  rightx_delx = np.zeros(len(idx[0]))
  righty_delx = np.zeros(len(idx[0]))
  rightz_delx = np.zeros(len(idx[0]))
  
  for ii in range(len(idx[0])):
    xx = idx[0][ii]
    yy = idx[1][ii]
    zz = idx[2][ii]
    pix = eigvecs[xx,yy,zz]
    left1 = eigvecs[xx-1,yy,zz]
    left2 = eigvecs[xx-2,yy,zz]

    if np.dot(pix, left1) < 0:
      left1 = -left1
    if np.dot(pix, left2) < 0:
      left2 = -left2
    
    #right_x = 0.5 * a[idx[0]-2, idx[1]] - 2.0 * a[idx[0]-1, idx[1]] + 1.5 * a[idx[0], idx[1]]
    rightx_delx[ii] = 0.5 * left2[0] - 2.0 * left1[0] + 1.5 * pix[0]
    righty_delx[ii] = 0.5 * left2[1] - 2.0 * left1[1] + 1.5 * pix[1]
    rightz_delx[ii] = 0.5 * left2[2] - 2.0 * left1[2] + 1.5 * pix[2]
  return(rightx_delx, righty_delx, rightz_delx)

def eigv_bottom_idx_3d(eigvecs, idx):
  # Return the gradient of each component of the eigenvector taken in the y direction
  # using only terms to the top
  # called bottom, because used for case when bottom direction (j-1) is out of bounds
  bottomx_dely = np.zeros(len(idx[0]))
  bottomy_dely = np.zeros(len(idx[0]))
  bottomz_dely = np.zeros(len(idx[0]))
  
  for ii in range(len(idx[0])):
    xx = idx[0][ii]
    yy = idx[1][ii]
    zz = idx[2][ii]
    pix = eigvecs[xx,yy,zz]
    top1 = eigvecs[xx,yy+1,zz]
    top2 = eigvecs[xx,yy+2,zz]

    if np.dot(pix, top1) < 0:
      top1 = -top1
    if np.dot(pix, top2) < 0:
      top2 = -top2
    
    #bottom_y = -0.5 * a[idx[0], idx[1]+2] + 2.0 * a[idx[0], idx[1]+1] - 1.5 * a[idx[0], idx[1]]
    bottomx_dely[ii] = -0.5 * top2[0] + 2.0 * top1[0] - 1.5 * pix[0]
    bottomy_dely[ii] = -0.5 * top2[1] + 2.0 * top1[1] - 1.5 * pix[1]
    bottomz_dely[ii] = -0.5 * top2[2] + 2.0 * top1[2] - 1.5 * pix[2]
  return(bottomx_dely, bottomy_dely, bottomz_dely)

def eigv_top_idx_3d(eigvecs, idx):
  # Return the gradient of each component of the eigenvector taken in the y direction
  # using only terms to the bottom
  # called top, because used for case when top direction (j+1) is out of bounds
  topx_dely = np.zeros(len(idx[0]))
  topy_dely = np.zeros(len(idx[0]))
  topz_dely = np.zeros(len(idx[0]))
  
  for ii in range(len(idx[0])):
    xx = idx[0][ii]
    yy = idx[1][ii]
    zz = idx[2][ii]
    pix = eigvecs[xx,yy,zz]
    bottom1 = eigvecs[xx,yy-1,zz]
    bottom2 = eigvecs[xx,yy-2,zz]

    if np.dot(pix, bottom1) < 0:
      bottom1 = -bottom1
    if np.dot(pix, bottom2) < 0:
      bottom2 = -bottom2
    
    #top_y = 0.5 * a[idx[0]+2, idx[1]] - 2.0 * a[idx[0]+1, idx[1]] + 1.5 * a[idx[0], idx[1]]
    topx_dely[ii] = 0.5 * bottom2[0] - 2.0 * bottom1[0] + 1.5 * pix[0]
    topy_dely[ii] = 0.5 * bottom2[1] - 2.0 * bottom1[1] + 1.5 * pix[1]
    topz_dely[ii] = 0.5 * bottom2[2] - 2.0 * bottom1[2] + 1.5 * pix[2]
  return(topx_dely, topy_dely, topz_dely)

def eigv_rear_idx_3d(eigvecs, idx):
  # Return the gradient of each component of the eigenvector taken in the z direction
  # using only terms to the front
  # called rear, because used for case when rear direction (k-1) is out of bounds
  rearx_delz = np.zeros(len(idx[0]))
  reary_delz = np.zeros(len(idx[0]))
  rearz_delz = np.zeros(len(idx[0]))
  
  for ii in range(len(idx[0])):
    xx = idx[0][ii]
    yy = idx[1][ii]
    zz = idx[2][ii]
    pix = eigvecs[xx,yy,zz]
    front1 = eigvecs[xx,yy,zz+1]
    front2 = eigvecs[xx,yy,zz+2]

    if np.dot(pix, front1) < 0:
      front1 = -front1
    if np.dot(pix, front2) < 0:
      front2 = -front2
    
    #rear_z = -0.5 * a[idx[0], idx[1], idx[2]+2] + 2.0 * a[idx[0], idx[1], idx[2]+1] - 1.5 * a[idx[0], idx[1], idx[2]]
    rearx_delz[ii] = -0.5 * front2[0] + 2.0 * front1[0] - 1.5 * pix[0]
    reary_delz[ii] = -0.5 * front2[1] + 2.0 * front1[1] - 1.5 * pix[1]
    rearz_delz[ii] = -0.5 * front2[2] + 2.0 * front1[2] - 1.5 * pix[2]
  return(rearx_delz, reary_delz, rearz_delz)

def eigv_front_idx_3d(eigvecs, idx):
  # Return the gradient of each component of the eigenvector taken in the z direction
  # using only terms to the rear
  # called front, because used for case when front direction (k+1) is out of bounds
  frontx_delz = np.zeros(len(idx[0]))
  fronty_delz = np.zeros(len(idx[0]))
  frontz_delz = np.zeros(len(idx[0]))
  
  for ii in range(len(idx[0])):
    xx = idx[0][ii]
    yy = idx[1][ii]
    zz = idx[2][ii]
    pix = eigvecs[xx,yy,zz]
    rear1 = eigvecs[xx,yy,zz-1]
    rear2 = eigvecs[xx,yy,zz-2]

    if np.dot(pix, rear1) < 0:
      rear1 = -rear1
    if np.dot(pix, rear2) < 0:
      rear2 = -rear2
    
    #front_z = 0.5 * a[idx[0]+2, idx[1]] - 2.0 * a[idx[0]+1, idx[1]] + 1.5 * a[idx[0], idx[1]]
    frontx_delz[ii] = 0.5 * rear2[0] - 2.0 * rear1[0] + 1.5 * pix[0]
    fronty_delz[ii] = 0.5 * rear2[1] - 2.0 * rear1[1] + 1.5 * pix[1]
    frontz_delz[ii] = 0.5 * rear2[2] - 2.0 * rear1[2] + 1.5 * pix[2]
  return(frontx_delz, fronty_delz, frontz_delz)


def run_test_2d():
  test_array = np.array([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20],[21,22,23,24,25]])
  test_x, test_y = fwd_diff_2d(test_array)
  print('test_array', test_array)
  print('fwd_diff_2d x', test_x)
  print('fwd_diff_2d y', test_y)
  test_x, test_y = back_diff_2d(test_array)
  print('back_diff_2d x', test_x)
  print('back_diff_2d y', test_y)
  print('left_diff_2d', left_diff_2d(test_array))
  print('right_diff_2d', right_diff_2d(test_array))
  print('bottom_diff_2d', bottom_diff_2d(test_array))
  print('top_diff_2d', top_diff_2d(test_array))

  tst_a = np.array([[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,1,2,3,4,5,0,0],[0,0,6,7,8,9,10,0,0],\
                  [0,0,11,12,13,14,15,0,0],[0,0,16,17,18,19,20,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0]])
  print('tst_a', tst_a)
  tst_da = np.zeros_like(tst_a)
  tst_da[np.where(tst_a[:,:] > 0)] = left_diff_idx_2d(tst_a, np.where(tst_a[:,:] > 0 ))
  print('left diff idx tst_a[np.where(tst_a[:,:] > 0)]\n', tst_da)
  tst_da[np.where(tst_a[:,:] > 0)] = right_diff_idx_2d(tst_a, np.where(tst_a[:,:] > 0 ))
  print('right diff idx tst_a[np.where(tst_a[:,:] > 0)]\n', tst_da)
  tst_da[np.where(tst_a[:,:] > 0)] = bottom_diff_idx_2d(tst_a, np.where(tst_a[:,:] > 0 ))
  print('bottom diff idx tst_a[np.where(tst_a[:,:] > 0)]\n', tst_da)
  tst_da[np.where(tst_a[:,:] > 0)] = top_diff_idx_2d(tst_a, np.where(tst_a[:,:] > 0 ))
  print('top diff idx tst_a[np.where(tst_a[:,:] > 0)]\n', tst_da)

def run_test_3d():
  test_array = np.array([[[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20],[21,22,23,24,25]],
                         [[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20],[21,22,23,24,25]],
                         [[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20],[21,22,23,24,25]],
                         [[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20],[21,22,23,24,25]],
                         [[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20],[21,22,23,24,25]]])
  
  test_x, test_y, test_z = fwd_diff_3d(test_array)
  print('test_array\n', test_array)
  print('fwd_diff_3d x\n', test_x)
  print('fwd_diff_3d y\n', test_y)
  print('fwd_diff_3d z\n', test_z)
  test_x, test_y, test_z = back_diff_3d(test_array)
  print('back_diff_3d x\n', test_x)
  print('back_diff_3d y\n', test_y)
  print('back_diff_3d z\n', test_z)
  print('left_diff_3d\n', left_diff_3d(test_array))
  print('right_diff_3d\n', right_diff_3d(test_array))
  print('bottom_diff_3d\n', bottom_diff_3d(test_array))
  print('top_diff_3d\n', top_diff_3d(test_array))
  print('rear_diff_3d\n', rear_diff_3d(test_array))
  print('front_diff_3d\n', front_diff_3d(test_array))

  tst_a = np.array([[[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0]],
                    [[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0]],
                    [[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,1,2,3,4,5,0,0],[0,0,6,7,8,9,10,0,0],
                     [0,0,11,12,13,14,15,0,0],[0,0,16,17,18,19,20,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0]],
                    [[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,1,2,3,4,5,0,0],[0,0,6,7,8,9,10,0,0],
                     [0,0,11,12,13,14,15,0,0],[0,0,16,17,18,19,20,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0]],
                    [[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,1,2,3,4,5,0,0],[0,0,6,7,8,9,10,0,0],
                     [0,0,11,12,13,14,15,0,0],[0,0,16,17,18,19,20,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0]],
                    [[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,1,2,3,4,5,0,0],[0,0,6,7,8,9,10,0,0],
                     [0,0,11,12,13,14,15,0,0],[0,0,16,17,18,19,20,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0]],
                    [[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,1,2,3,4,5,0,0],[0,0,6,7,8,9,10,0,0],
                     [0,0,11,12,13,14,15,0,0],[0,0,16,17,18,19,20,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0]],
                    [[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0]],
                    [[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0]]])
  print('tst_a\n', tst_a)
  tst_da = np.zeros_like(tst_a)
  tst_da[np.where(tst_a[:,:] > 0)] = left_diff_idx_3d(tst_a, np.where(tst_a[:,:] > 0 ))
  print('left diff idx tst_a[np.where(tst_a[:,:] > 0)]\n', tst_da)
  tst_da = np.zeros_like(tst_a)
  tst_da[np.where(tst_a[:,:] > 0)] = right_diff_idx_3d(tst_a, np.where(tst_a[:,:] > 0 ))
  print('right diff idx tst_a[np.where(tst_a[:,:] > 0)]\n', tst_da)
  tst_da = np.zeros_like(tst_a)
  tst_da[np.where(tst_a[:,:] > 0)] = bottom_diff_idx_3d(tst_a, np.where(tst_a[:,:] > 0 ))
  print('bottom diff idx tst_a[np.where(tst_a[:,:] > 0)]\n', tst_da)
  tst_da = np.zeros_like(tst_a)
  tst_da[np.where(tst_a[:,:] > 0)] = top_diff_idx_3d(tst_a, np.where(tst_a[:,:] > 0 ))
  print('top diff idx tst_a[np.where(tst_a[:,:] > 0)]\n', tst_da)
  tst_da = np.zeros_like(tst_a)
  tst_da[np.where(tst_a[:,:] > 0)] = rear_diff_idx_3d(tst_a, np.where(tst_a[:,:] > 0 ))
  print('rear diff idx tst_a[np.where(tst_a[:,:] > 0)]\n', tst_da)
  tst_da = np.zeros_like(tst_a)
  tst_da[np.where(tst_a[:,:] > 0)] = front_diff_idx_3d(tst_a, np.where(tst_a[:,:] > 0 ))
  print('front diff idx tst_a[np.where(tst_a[:,:] > 0)]\n', tst_da)
  
if __name__ == "__main__":
  run_test_3d()
