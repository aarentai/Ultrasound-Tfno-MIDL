from lazy_imports import np, ndimage
from data.convert import get_framework

def is_interior_pt_2d(ii, jj, mask):
  if ((ii == 0) or (ii == mask.shape[0]-1) or
      (jj == 0) or (jj == mask.shape[1]-1)):
    return(False)
  return(( mask[ii-1,jj-1] + mask[ii-1,jj] + mask[ii-1,jj+1] \
         + mask[ii,jj-1] + mask[ii,jj] + mask[ii,jj+1] \
         + mask[ii+1,jj-1] + mask[ii+1,jj] + mask[ii+1,jj+1]) == 9)


def is_interior_pt_3d(ii, jj, kk, mask):
  if ((ii < 1) or (ii > mask.shape[0]-2) or
      (jj < 1) or (jj > mask.shape[1]-2) or
      (kk < 1) or (kk > mask.shape[2]-2)):
    return(False)
  return(( mask[ii-1,jj-1,kk-1] + mask[ii-1,jj,kk-1] + mask[ii-1,jj+1,kk-1] \
           + mask[ii,jj-1,kk-1] + mask[ii,jj,kk-1] + mask[ii,jj+1,kk-1] \
           + mask[ii+1,jj-1,kk-1] + mask[ii+1,jj,kk-1] + mask[ii+1,jj+1,kk-1] \
           + mask[ii-1,jj-1,kk] + mask[ii-1,jj,kk] + mask[ii-1,jj+1,kk] \
           + mask[ii,jj-1,kk] + mask[ii,jj,kk] + mask[ii,jj+1,kk] \
           + mask[ii+1,jj-1,kk] + mask[ii+1,jj,kk] + mask[ii+1,jj+1,kk] \
           + mask[ii-1,jj-1,kk+1] + mask[ii-1,jj,kk+1] + mask[ii-1,jj+1,kk+1] \
           + mask[ii,jj-1,kk+1] + mask[ii,jj,kk+1] + mask[ii,jj+1,kk+1] \
           + mask[ii+1,jj-1,kk+1] + mask[ii+1,jj,kk+1] + mask[ii+1,jj+1,kk+1]) == 27)

def is_3_in_a_row_3d(ii, jj, kk, mask, axis=0):
  if axis == 0:
    return((ii > 0) and (ii < mask.shape[0]-1) and
           (jj >= 0) and (jj <= mask.shape[1]-1) and
           (kk >= 0) and (kk <= mask.shape[2]-1) and
           (mask[ii-1,jj,kk] + mask[ii,jj,kk] + mask[ii+1,jj,kk] == 3))
  elif axis == 1:
    return((ii >= 0) and (ii <= mask.shape[0]-1) and
           (jj > 0) and (jj < mask.shape[1]-1) and
           (kk >= 0) and (kk <= mask.shape[2]-1) and
           (mask[ii,jj-1,kk] + mask[ii,jj,kk] + mask[ii,jj+1,kk] == 3))
  else: #axis == 2
    return((ii >= 0) and (ii <= mask.shape[0]-1) and
           (jj >= 0) and (jj <= mask.shape[1]-1) and
           (kk > 0) and (kk < mask.shape[2]-1) and
           (mask[ii,jj,kk-1] + mask[ii,jj,kk] + mask[ii,jj,kk+1] == 3))

def is_offset_in_mask_3d(ii, jj, kk, offs, mask):
  in_mask = True
  for off in offs:
    iioff = ii+off[0]
    jjoff = jj+off[1]
    kkoff = kk+off[2]
    if ((iioff < 0) or (iioff > mask.shape[0]-1) or
      (jjoff < 0) or (jjoff > mask.shape[1]-1) or
      (kkoff < 0) or (kkoff > mask.shape[2]-1)):
      in_mask = False
      break
    if mask[iioff,jjoff,kkoff] == 0:
      in_mask = False
      break
  return(in_mask)

def determine_boundary_2d(mask):
  # Classify boundary points of mask
  # WARNING!  This method modifies the mask as well, by setting values to 0 for any boundary type
  #  that cannot be differentiated accurately (currently).
  # Returns
  #   bdry_type - a label image labelled with boundary type
  #   bdry_idx - a dictionary whose values contain the indices for pixels with the key's boundary type
  #   bdry_map - a dictionary mapping the boundary type name to its associated value
  #
  #   mask - values are set to 0 for any boundary type we cannot currently differentiate
  fw, fw_name = get_framework(mask)
  xsz = mask.shape[0]
  ysz = mask.shape[1]
  bdry_type = fw.zeros_like(mask)
  bdry_idx = {}
  bdry_map = {}
  bdry_map["outside"] = 0
  bdry_map["interior"] = 1
  bdry_map["interiorleft"] = 2
  bdry_map["interiorright"] = 3
  bdry_map["interiorbottom"] = 4
  bdry_map["interiortop"] = 5
  bdry_map["left"] = 6
  bdry_map["right"] = 7
  bdry_map["bottom"] = 8
  bdry_map["top"] = 9
  bdry_map["bottomleft"] = 10
  bdry_map["topleft"] = 11
  bdry_map["bottomright"] = 12
  bdry_map["topright"] = 13
  bdry_map["notleft"] = 14
  bdry_map["notright"] = 15
  bdry_map["notbottom"] = 16
  bdry_map["nottop"] = 17

  for ii in range(xsz):
    for jj in range(ysz):
      if mask[ii,jj]:
        if is_interior_pt_2d(ii, jj, mask):
          # interior point
          bdry_type[ii,jj] = bdry_map["interior"]

        elif (mask[ii-1, jj] and mask[ii+1, jj] and mask[ii, jj-1] and mask[ii, jj+1]):
          # mostly interior, especially for first deriv purposes.
          # counts only as boundary for second derivative cross term calculations d2/dxy etc.
          if mask[ii+1,jj-1] and mask[ii+1,jj+1]:
            # interior left
            bdry_type[ii,jj] = bdry_map["interiorleft"]

          elif mask[ii-1,jj-1] and mask[ii-1,jj+1]:
            # interior right
            bdry_type[ii,jj] = bdry_map["interiorright"]

          elif mask[ii-1,jj+1] and mask[ii+1,jj+1]:
            # interior bottom
            bdry_type[ii,jj] = bdry_map["interiorbottom"]

          elif mask[ii-1,jj-1] and mask[ii+1,jj-1]:
            # interior top
            bdry_type[ii,jj] = bdry_map["interiortop"]

        elif is_interior_pt_2d(ii+1, jj, mask):
          # left
          bdry_type[ii,jj] = bdry_map["left"]

        elif is_interior_pt_2d(ii-1, jj, mask):
          # right
          bdry_type[ii,jj] = bdry_map["right"]

        elif is_interior_pt_2d(ii, jj+1, mask):
          # bottom
          bdry_type[ii,jj] = bdry_map["bottom"]

        elif is_interior_pt_2d(ii, jj-1, mask):
          # top
          bdry_type[ii,jj] = bdry_map["top"]
          
        elif is_interior_pt_2d(ii+1, jj+1, mask):
          # left bottom
          bdry_type[ii,jj] = bdry_map["bottomleft"]

        elif is_interior_pt_2d(ii+1, jj-1, mask):
          # left top
          bdry_type[ii,jj] = bdry_map["topleft"]

        elif is_interior_pt_2d(ii-1, jj+1, mask):
          # right bottom
          bdry_type[ii,jj] = bdry_map["bottomright"]

        elif is_interior_pt_2d(ii-1, jj-1, mask):
          # right top
          bdry_type[ii,jj] = bdry_map["topright"]
          
        elif (mask[ii-1, jj-1] and mask[ii-1, jj+1] and mask[ii-1, jj] and mask[ii-2, jj]):
          # not left
          # TODO decide whether better to mask or not mask these pixels
          # If we set mask to 0 here, then we need to set new boundary pixels accordingly!
          #print("notleft masking pixel", ii, jj)
          bdry_type[ii,jj] = bdry_map["notleft"]
          #mask[ii,jj] = 0

        elif (mask[ii+1, jj-1] and mask[ii+1, jj+1] and mask[ii+1, jj] and mask[ii+2, jj]):
          # not right
          #print("notright masking pixel", ii, jj)
          bdry_type[ii,jj] = bdry_map["notright"]
          #mask[ii,jj] = 0

        elif (mask[ii-1, jj-1] and mask[ii+1, jj-1] and mask[ii, jj-1] and mask[ii, jj-2]):
          # not bottom
          #print("notbottom masking pixel", ii, jj)
          bdry_type[ii,jj] = bdry_map["notbottom"]
          #mask[ii,jj] = 0

        elif (mask[ii-1, jj+1] and mask[ii+1, jj+1] and mask[ii, jj+1] and mask[ii, jj+2]):
          # not top
          #print("nottop masking pixel", ii, jj)
          bdry_type[ii,jj] = bdry_map["nottop"]
          #mask[ii,jj] = 0

        else:
          # can't take derivatives currently
          # Note that with care there are some cases here where we still can take derivatives
          # just not as currently implemented.  For now, mask these regions
          print("masking pixel", ii, jj)
          mask[ii,jj] = 0
          bdry_type[ii,jj] = 0
        
      else:
        bdry_type[ii,jj] = 0

  for ii in range(1,18):
    bdry_idx[ii] = fw.where(bdry_type == ii)

  return(bdry_type, bdry_idx, bdry_map)
# end determine_boundary_2d

def determine_boundary_3d(mask, modify_mask=True):
  # Classify boundary points of mask
  # WARNING!  This method modifies the mask as well, by setting values to 0 for any boundary type
  #  that cannot be differentiated accurately (currently).
  # Returns
  #   bdry_type - a label image labelled with boundary type
  #   bdry_idx - a dictionary whose values contain the indices for pixels with the key's boundary type
  #   bdry_map - a dictionary mapping the boundary type name to its associated value
  #
  #   mask - values are set to 0 for any boundary type we cannot currently differentiate
  fw, fw_name = get_framework(mask)
  xsz = mask.shape[0]
  ysz = mask.shape[1]
  zsz = mask.shape[2]
  bdry_type = fw.zeros_like(mask)
  bdry_idx = {}
  bdry_map = {}
  bdry_map["outside"] = 0
  bdry_map["interior"] = 1
  # num_known_bdrys = 2
  bdry_map["interiorleft"] = 2
  bdry_map["interiorright"] = 3
  bdry_map["interiorbottom"] = 4
  bdry_map["interiortop"] = 5
  bdry_map["interiorrear"] = 6
  bdry_map["interiorfront"] = 7
  num_known_bdrys = 8

  # TODO Delete below, note that numbering is off and also auto determined from following code
  # bdry_map["left"] = 6
  # bdry_map["right"] = 7
  # bdry_map["bottom"] = 8
  # bdry_map["top"] = 9
  # bdry_map["bottomleft"] = 10
  # bdry_map["topleft"] = 11
  # bdry_map["bottomright"] = 12
  # bdry_map["topright"] = 13
  # bdry_map["shiftyleft"] = 14
  # bdry_map["shiftyright"] = 15
  # bdry_map["bottomshiftx"] = 16
  # bdry_map["topshiftx"] = 17
  # bdry_map["rear"] = 20
  # bdry_map["front"] = 21
  # bdry_map["rearleft"] = 22
  # bdry_map["frontleft"] = 23
  # bdry_map["rearright"] = 24
  # bdry_map["frontright"] = 25
  # bdry_map["rearbottom"] = 26
  # bdry_map["frontbottom"] = 27
  # bdry_map["reartop"] = 28
  # bdry_map["fronttop"] = 29
  # bdry_map["rearshiftx"] = 30
  # bdry_map["frontshiftx"] = 31
  # bdry_map["rearshifty"] = 32
  # bdry_map["frontshifty"] = 33
  # bdry_map["shiftzleft"] = 34
  # bdry_map["shiftzright"] = 35
  # bdry_map["shiftzbottom"] = 36
  # bdry_map["shiftztop"] = 37
  # bdry_map["rearbottomleft"] = 38
  # bdry_map["frontbottomleft"] = 39
  # bdry_map["rearbottomright"] = 40
  # bdry_map["frontbottomright"] = 41
  # bdry_map["reartopleft"] = 42
  # bdry_map["fronttopleft"] = 43
  # bdry_map["reartopright"] = 44
  # bdry_map["fronttopright"] = 45
  # bdry_map["rearshiftyleft"] = 46
  # bdry_map["rearshiftyright"] = 47
  # bdry_map["frontshiftyleft"] = 48
  # bdry_map["frontshiftyright"] = 49
  # bdry_map["shiftzshiftyleft"] = 50
  # bdry_map["shiftzshiftyright"] = 51
  # bdry_map["rearbottomshiftx"] = 52
  # bdry_map["reartopshiftx"] = 53
  # bdry_map["frontbottomshiftx"] = 54
  # bdry_map["fronttopshiftx"] = 55
  # bdry_map["shiftzbottomshiftx"] = 56
  # bdry_map["shiftztopshiftx"] = 57
  # bdry_map["shiftzbottomleft"] = 58
  # bdry_map["shiftzbottomright"] = 59
  # bdry_map["shiftztopleft"] = 60
  # bdry_map["shiftztopright"] = 61

  num_red = 0
  num_no = 0
  
  for ii in range(xsz):
    for jj in range(ysz):
      for kk in range(zsz):
        if mask[ii,jj,kk]:
          if is_interior_pt_3d(ii, jj, kk, mask):
            # interior point
            bdry_type[ii,jj,kk] = bdry_map["interior"]
        
          elif is_offset_in_mask_3d(ii, jj, kk, [[-1,0,0],[1,0,0],[0,-1,0],[0,1,0],[0,0,-1],[0,0,1]], mask):
                # mostly interior, especially for first deriv purposes.
            # counts only as boundary for second derivative cross term calculations d2/dxy etc.
            if is_offset_in_mask_3d(ii, jj, kk, [[1,-1,0],[1,1,0],[1,0,-1],[1,0,1]], mask):
              # interior left
              bdry_type[ii,jj,kk] = bdry_map["interiorleft"]
        
            elif is_offset_in_mask_3d(ii, jj, kk, [[-1,-1,0],[-1,1,0],[-1,0,-1],[-1,0,1]], mask):
              # interior right
              bdry_type[ii,jj,kk] = bdry_map["interiorright"]
        
            elif is_offset_in_mask_3d(ii, jj, kk, [[-1,1,0],[1,1,0],[0,1,-1],[0,1,1]], mask):
              # interior bottom
              bdry_type[ii,jj,kk] = bdry_map["interiorbottom"]
        
            elif is_offset_in_mask_3d(ii, jj, kk, [[-1,-1,0],[1,-1,0],[0,-1,-1],[0,-1,1]], mask):
              # interior top
              bdry_type[ii,jj,kk] = bdry_map["interiortop"]

            elif is_offset_in_mask_3d(ii, jj, kk, [[-1,0,1],[1,0,1],[0,-1,1],[0,1,1]], mask):
              # interior rear
              bdry_type[ii,jj,kk] = bdry_map["interiorrear"]
        
            elif is_offset_in_mask_3d(ii, jj, kk, [[-1,0,-1],[1,0,-1],[0,-1,-1],[0,1,-1]], mask):
              # interior front
              bdry_type[ii,jj,kk] = bdry_map["interiorfront"]
           # bdry_name = ''
           # if mask[ii,jj+1,kk+1]:
           #   bdry_name += "interiorrearbottom"
           # elif mask[ii,jj-1,kk+1]:
           #   bdry_name += "interiorreartop"
           # elif mask[ii,jj+1,kk-1]:
           #   bdry_name += "interiorfrontbottom"
           # elif mask[ii,jj-1,kk-1]:
           #   bdry_name += "interiorfronttop"
           # else:
           #   print('unexpected interior case A at', ii, jj, kk)
           # if mask[ii+1,jj,kk+1]:
           #   bdry_name += "interiorrearleft"
           # elif mask[ii-1,jj,kk+1]:
           #   bdry_name += "interiorrearright"
           # elif mask[ii+1,jj,kk-1]:
           #   bdry_name += "interiorfrontleft"
           # elif mask[ii-1,jj,kk-1]:
           #   bdry_name += "interiorfrontright"
           # else:
           #   print('unexpected interior case B at', ii, jj, kk)
           # if mask[ii+1,jj+1,kk]: 
           #   bdry_name += "interiorbottomleft"
           # elif mask[ii-1,jj+1,kk]: 
           #   bdry_name += "interiorbottomright"
           # elif mask[ii+1,jj-1,kk]: 
           #   bdry_name += "interiortopleft"
           # elif mask[ii-1,jj-1,kk]: 
           #   bdry_name += "interiortopright"
           # else:
           #   print('unexpected interior case C at', ii, jj, kk)
           # # if (mask[ii,jj-1,kk+1] and mask[ii,jj+1,kk-1]):
           # #   bdry_name += "interiorreartop" # same as frontbottom
           # # elif (mask[ii,jj+1,kk+1] and mask[ii,jj-1,kk-1]):
           # #   bdry_name += "interiorrearbottom" # same as fronttop
           # # elif (mask[ii,jj+1,kk+1] and mask[ii,jj+1,kk-1]): 
           # #   bdry_name += "interiorzbottom" 
           # # elif (mask[ii,jj+1,kk+1] and mask[ii,jj-1,kk+1]): 
           # #   bdry_name += "interiorreary"
           # # elif (mask[ii,jj-1,kk-1] and mask[ii,jj-1,kk+1]): 
           # #   bdry_name += "interiorztop"
           # # elif (mask[ii,jj-1,kk-1] and mask[ii,jj+1,kk-1]): 
           # #   bdry_name += "interiorfronty" 
           # # else:
           # #   print('unexpected interior case A at', ii, jj, kk)
           # # if (mask[ii-1,jj,kk+1] and mask[ii+1,jj,kk-1]):
           # #   bdry_name += "interiorrearright" # same as frontleft
           # # elif (mask[ii+1,jj,kk+1] and mask[ii-1,jj,kk-1]):
           # #   bdry_name += "interiorrearleft" # same as frontright
           # # elif (mask[ii+1,jj,kk+1] and mask[ii+1,jj,kk-1]): 
           # #   bdry_name += "interiorzleft" 
           # # elif (mask[ii+1,jj,kk+1] and mask[ii-1,jj,kk+1]): 
           # #   bdry_name += "interiorrearx"
           # # elif (mask[ii-1,jj,kk-1] and mask[ii-1,jj,kk+1]): 
           # #   bdry_name += "interiorzright"
           # # elif (mask[ii-1,jj,kk-1] and mask[ii+1,jj,kk-1]): 
           # #   bdry_name += "interiorfrontx" 
           # # else:
           # #   print('unexpected interior case B at', ii, jj, kk)
           # # if (mask[ii+1,jj-1,kk] and mask[ii-1,jj+1,kk]): 
           # #   bdry_name += "interiorbottomright" # same as topleft
           # # elif (mask[ii+1,jj+1,kk] and mask[ii-1,jj-1,kk]): 
           # #   bdry_name += "interiorbottomleft" # same as topright
           # # elif (mask[ii+1,jj+1,kk] and mask[ii+1,jj-1,kk]): 
           # #   bdry_name += "interioryleft" 
           # # elif (mask[ii+1,jj+1,kk] and mask[ii-1,jj+1,kk]): 
           # #   bdry_name += "interiorbottomx"
           # # elif (mask[ii-1,jj-1,kk] and mask[ii-1,jj+1,kk]): 
           # #   bdry_name += "interioryright"
           # # elif (mask[ii-1,jj-1,kk] and mask[ii+1,jj-1,kk]): 
           # #   bdry_name += "interiortopx" 
           # # else:
           # #   print('unexpected interior case C at', ii, jj, kk)
           # if bdry_name not in bdry_map:
           #   bdry_map[bdry_name] = num_known_bdrys
           #   num_known_bdrys += 1
           # bdry_type[ii,jj,kk] = bdry_map[bdry_name]     
           
        
          else:
            # build things up, z to x direction
            bdry_name = ''
            if is_3_in_a_row_3d(ii, jj, kk, mask, 2):
              # interior pt z dir
              bdry_name += 'intz'
            elif is_3_in_a_row_3d(ii, jj, kk+1, mask, 2):
              bdry_name += 'rear'
            elif is_3_in_a_row_3d(ii, jj, kk-1, mask, 2):
              bdry_name += 'front'
            elif is_3_in_a_row_3d(ii+1, jj, kk, mask, 2) and is_3_in_a_row_3d(ii+2, jj, kk, mask, 2):
              if is_offset_in_mask_3d(ii, jj, kk, [[3,0,0],[3,0,-1]], mask):
                # Can do more accurate 2nd derivative
                bdry_name += 'shiftztol'
              else:
                # Must do reduced accuracy 2nd derivative
                bdry_name += 'shiftztolred'
            elif is_3_in_a_row_3d(ii-1, jj, kk, mask, 2) and is_3_in_a_row_3d(ii-2, jj, kk, mask, 2):
              if is_offset_in_mask_3d(ii, jj, kk, [[-3,0,0],[-3,0,-1]], mask):
                bdry_name += 'shiftztori'
              else:
                bdry_name += 'shiftztorired'
            elif is_3_in_a_row_3d(ii, jj+1, kk, mask, 2) and is_3_in_a_row_3d(ii, jj+2, kk, mask, 2):
              if is_offset_in_mask_3d(ii, jj, kk, [[0,3,0],[0,3,-1]], mask):
                bdry_name += 'shiftztob'
              else:
                bdry_name += 'shiftztobred'
            elif is_3_in_a_row_3d(ii, jj-1, kk, mask, 2) and is_3_in_a_row_3d(ii, jj-2, kk, mask, 2):
              if is_offset_in_mask_3d(ii, jj, kk, [[-3,0,0],[-3,0,-1]], mask):
                bdry_name += 'shiftztot'
              else:
                bdry_name += 'shiftztotred'
            #elif is_3_in_a_row_3d(ii+1, jj, kk, mask, 0) and is_3_in_a_row_3d(ii+1, jj, kk+1, mask, 2):
            elif is_offset_in_mask_3d(ii, jj, kk, [[1,0,1],[1,0,2],[2,0,1],[2,0,0],[1,0,0]], mask):
              if is_offset_in_mask_3d(ii, jj, kk, [[2,0,2],[3,0,0],[3,0,1]], mask):
                bdry_name += 'shiftreartol'
              else:
                bdry_name += 'shiftreartolred'
            #elif is_3_in_a_row_3d(ii-1, jj, kk, mask, 0) and is_3_in_a_row_3d(ii-1, jj, kk+1, mask, 2):
            elif is_offset_in_mask_3d(ii, jj, kk, [[-1,0,1],[-1,0,2],[-2,0,1],[-2,0,0],[-1,0,0]], mask):
              if is_offset_in_mask_3d(ii, jj, kk, [[-2,0,2],[-3,0,0],[-3,0,1]], mask):
                bdry_name += 'shiftreartori'
              else:
                bdry_name += 'shiftreartorired'
            #elif is_3_in_a_row_3d(ii, jj+1, kk, mask, 1) and is_3_in_a_row_3d(ii, jj+1, kk+1, mask, 2):
            elif is_offset_in_mask_3d(ii, jj, kk, [[0,1,1],[0,1,2],[0,2,1],[0,2,0],[0,1,0]], mask):
              if is_offset_in_mask_3d(ii, jj, kk, [[0,2,2],[0,3,0],[0,3,1]], mask):
                bdry_name += 'shiftreartob'
              else:
                bdry_name += 'shiftreartobred'
            #elif is_3_in_a_row_3d(ii, jj-1, kk, mask, 1) and is_3_in_a_row_3d(ii, jj-1, kk+1, mask, 2):
            elif is_offset_in_mask_3d(ii, jj, kk, [[0,-1,1],[0,-1,2],[0,-2,1],[0,-2,0],[0,-1,0]], mask):
              if is_offset_in_mask_3d(ii, jj, kk, [[0,-2,2],[0,-3,0],[0,-3,1]], mask):
                bdry_name += 'shiftreartot'
              else:
                bdry_name += 'shiftreartotred'
            #elif is_3_in_a_row_3d(ii+1, jj, kk, mask, 0) and is_3_in_a_row_3d(ii+1, jj, kk-1, mask, 2):
            elif is_offset_in_mask_3d(ii, jj, kk, [[1,0,-1],[1,0,-2],[2,0,-1],[2,0,0],[1,0,0]], mask):
              if is_offset_in_mask_3d(ii, jj, kk, [[2,0,-2],[3,0,0],[3,0,-1]], mask):
                bdry_name += 'shiftfronttol'
              else:
                bdry_name += 'shiftfronttolred'
            #elif is_3_in_a_row_3d(ii-1, jj, kk, mask, 0) and is_3_in_a_row_3d(ii-1, jj, kk-1, mask, 2):
            elif is_offset_in_mask_3d(ii, jj, kk, [[-1,0,-1],[-1,0,-2],[-2,0,-1],[-2,0,0],[-1,0,0]], mask):
              if is_offset_in_mask_3d(ii, jj, kk, [[-2,0,-2],[-3,0,0],[-3,0,-1]], mask):
                bdry_name += 'shiftfronttori'
              else:
                bdry_name += 'shiftfronttorired'
            #elif is_3_in_a_row_3d(ii, jj+1, kk, mask, 1) and is_3_in_a_row_3d(ii, jj+1, kk-1, mask, 2):
            elif is_offset_in_mask_3d(ii, jj, kk, [[0,1,-1],[0,1,-2],[0,2,-1],[0,2,0],[0,1,0]], mask):
              if is_offset_in_mask_3d(ii, jj, kk, [[0,2,-2],[0,3,0],[0,3,-1]], mask):
                bdry_name += 'shiftfronttob'
              else:
                bdry_name += 'shiftfronttobred'
            #elif is_3_in_a_row_3d(ii, jj-1, kk, mask, 1) and is_3_in_a_row_3d(ii, jj-1, kk-1, mask, 2):
            elif is_offset_in_mask_3d(ii, jj, kk, [[0,-1,-1],[0,-1,-2],[0,-2,-1],[0,-2,0],[0,-1,0]], mask):
              if is_offset_in_mask_3d(ii, jj, kk, [[0,-2,-2],[0,-3,0],[0,-3,-1]], mask):
                bdry_name += 'shiftfronttot'
              else:
                bdry_name += 'shiftfronttotred'
            else:
              bdry_name += 'noz'
            if is_3_in_a_row_3d(ii, jj, kk, mask, 1):
              # interior pt y dir
              bdry_name += 'inty'
            elif is_3_in_a_row_3d(ii, jj+1, kk, mask, 1):
              bdry_name += 'bottom'
            elif is_3_in_a_row_3d(ii, jj-1, kk, mask, 1):
              bdry_name += 'top'
            elif is_3_in_a_row_3d(ii+1, jj, kk, mask, 1) and is_3_in_a_row_3d(ii+2, jj, kk, mask, 1):
              if is_offset_in_mask_3d(ii, jj, kk, [[3,0,0],[3,-1,0]], mask):
                bdry_name += 'shiftytol'
              else:
                bdry_name += 'shiftytolred'
            elif is_3_in_a_row_3d(ii-1, jj, kk, mask, 1) and is_3_in_a_row_3d(ii-2, jj, kk, mask, 1):
              if is_offset_in_mask_3d(ii, jj, kk, [[-3,0,0],[-3,-1,0]], mask):
                bdry_name += 'shiftytori'
              else:
                bdry_name += 'shiftytorired'
            elif is_3_in_a_row_3d(ii, jj, kk+1, mask, 1) and is_3_in_a_row_3d(ii, jj, kk+2, mask, 1):
              if is_offset_in_mask_3d(ii, jj, kk, [[0,0,3],[0,-1,3]], mask):
                bdry_name += 'shiftytore'
              else:
                bdry_name += 'shiftytorered'
            elif is_3_in_a_row_3d(ii, jj, kk-1, mask, 1) and is_3_in_a_row_3d(ii, jj, kk-2, mask, 1):
              if is_offset_in_mask_3d(ii, jj, kk, [[0,0,-3],[0,-1,-3]], mask):
                bdry_name += 'shiftytof'
              else:
                bdry_name += 'shiftytofred'
            #elif is_3_in_a_row_3d(ii+1, jj, kk, mask, 0) and is_3_in_a_row_3d(ii+1, jj+1, kk, mask, 1):
            elif is_offset_in_mask_3d(ii, jj, kk, [[1,1,0],[1,2,0],[2,1,0],[2,0,0],[1,0,0]], mask):
              if is_offset_in_mask_3d(ii, jj, kk, [[2,2,0],[3,0,0],[3,1,0]], mask):
                bdry_name += 'shiftbottomtol'
              else:
                bdry_name += 'shiftbottomtolred'
            #elif is_3_in_a_row_3d(ii-1, jj, kk, mask, 0) and is_3_in_a_row_3d(ii-1, jj+1, kk, mask, 1):
            elif is_offset_in_mask_3d(ii, jj, kk, [[-1,1,0],[-1,2,0],[-2,1,0],[-2,0,0],[-1,0,0]], mask):
              if is_offset_in_mask_3d(ii, jj, kk, [[-2,2,0],[-3,0,0],[-3,1,0]], mask):
                bdry_name += 'shiftbottomtori'
              else:
                bdry_name += 'shiftbottomtorired'
            #elif is_3_in_a_row_3d(ii, jj, kk+1, mask, 2) and is_3_in_a_row_3d(ii, jj+1, kk+1, mask, 1):
            elif is_offset_in_mask_3d(ii, jj, kk, [[0,1,1],[0,2,1],[0,1,2],[0,0,2],[0,0,1]], mask):
              if is_offset_in_mask_3d(ii, jj, kk, [[0,2,2],[0,0,3],[0,1,3]], mask):
                bdry_name += 'shiftbottomtore'
              else:
                bdry_name += 'shiftbottomtorered'
            #elif is_3_in_a_row_3d(ii, jj, kk-1, mask, 2) and is_3_in_a_row_3d(ii, jj+1, kk-1, mask, 1):
            elif is_offset_in_mask_3d(ii, jj, kk, [[0,1,-1],[0,2,-1],[0,1,-2],[0,0,-2],[0,0,-1]], mask):
              if is_offset_in_mask_3d(ii, jj, kk, [[0,2,-2],[0,0,-3],[0,1,-3]], mask):
                bdry_name += 'shiftbottomtof'
              else:
                bdry_name += 'shiftbottomtofred'
            #elif is_3_in_a_row_3d(ii+1, jj, kk, mask, 0) and is_3_in_a_row_3d(ii+1, jj-1, kk, mask, 1):
            elif is_offset_in_mask_3d(ii, jj, kk, [[1,-1,0],[1,-2,0],[2,-1,0],[2,0,0],[1,0,0]], mask):
              if is_offset_in_mask_3d(ii, jj, kk, [[2,-2,0],[3,0,0],[3,-1,0]], mask):
                bdry_name += 'shifttoptol'
              else:
                bdry_name += 'shifttoptolred'
            #elif is_3_in_a_row_3d(ii-1, jj, kk, mask, 0) and is_3_in_a_row_3d(ii-1, jj-1, kk, mask, 1):
            elif is_offset_in_mask_3d(ii, jj, kk, [[-1,-1,0],[-1,-2,0],[-2,-1,0],[-2,0,0],[-1,0,0]], mask):
              if is_offset_in_mask_3d(ii, jj, kk, [[-2,-2,0],[-3,0,0],[-3,-1,0]], mask):
                bdry_name += 'shifttoptori'
              else:
                bdry_name += 'shifttoptorired'
            #elif is_3_in_a_row_3d(ii, jj, kk+1, mask, 2) and is_3_in_a_row_3d(ii, jj-1, kk+1, mask, 1):
            elif is_offset_in_mask_3d(ii, jj, kk, [[0,-1,1],[0,-2,1],[0,-1,2],[0,0,2],[0,0,1]], mask):
              if is_offset_in_mask_3d(ii, jj, kk, [[0,-2,2],[0,0,3],[0,-1,3]], mask):
                bdry_name += 'shifttoptore'
              else:
                bdry_name += 'shifttoptorered'
            #elif is_3_in_a_row_3d(ii, jj, kk-1, mask, 2) and is_3_in_a_row_3d(ii, jj-1, kk-1, mask, 1):
            elif is_offset_in_mask_3d(ii, jj, kk, [[0,-1,-1],[0,-2,-1],[0,-1,-2],[0,0,-2],[0,0,-1]], mask):
              if is_offset_in_mask_3d(ii, jj, kk, [[0,-2,-2],[0,0,-3],[0,-1,-3]], mask):
                bdry_name += 'shifttoptof'
              else:
                bdry_name += 'shifttoptofred'
            else:
              bdry_name += 'noy'
            if is_3_in_a_row_3d(ii, jj, kk, mask, 0):
              # interior pt x dir
              bdry_name += 'intx'
            elif is_3_in_a_row_3d(ii+1, jj, kk, mask, 0):
              bdry_name += 'left'
            elif is_3_in_a_row_3d(ii-1, jj, kk, mask, 0):
              bdry_name += 'right'
            elif is_3_in_a_row_3d(ii, jj+1, kk, mask, 0) and is_3_in_a_row_3d(ii, jj+2, kk, mask, 0):
              if is_offset_in_mask_3d(ii, jj, kk, [[0,3,0],[-1,3,0]], mask):
                bdry_name += 'shiftxtob'
              else:
                bdry_name += 'shiftxtobred'
            elif is_3_in_a_row_3d(ii, jj-1, kk, mask, 0) and is_3_in_a_row_3d(ii, jj-2, kk, mask, 0):
              if is_offset_in_mask_3d(ii, jj, kk, [[0,-3,0],[-1,-3,0]], mask):
                bdry_name += 'shiftxtot'
              else:
                bdry_name += 'shiftxtotred'
            elif is_3_in_a_row_3d(ii, jj, kk+1, mask, 0) and is_3_in_a_row_3d(ii, jj, kk+2, mask, 0):
              if is_offset_in_mask_3d(ii, jj, kk, [[0,0,3],[-1,0,3]], mask):
                bdry_name += 'shiftxtore'
              else:
                bdry_name += 'shiftxtorered'
            elif is_3_in_a_row_3d(ii, jj, kk-1, mask, 0) and is_3_in_a_row_3d(ii, jj, kk-2, mask, 0):
              if is_offset_in_mask_3d(ii, jj, kk, [[0,0,-3],[-1,0,-3]], mask):
                bdry_name += 'shiftxtof'
              else:
                bdry_name += 'shiftxtofred'
            #elif is_3_in_a_row_3d(ii, jj+1, kk, mask, 1) and is_3_in_a_row_3d(ii+1, jj+1, kk, mask, 0):
            elif is_offset_in_mask_3d(ii, jj, kk, [[1,1,0],[2,1,0],[1,2,0],[0,2,0],[0,1,0]], mask):
              if is_offset_in_mask_3d(ii, jj, kk, [[2,2,0],[0,3,0],[1,3,0]], mask):
                bdry_name += 'shiftlefttob'
              else:
                bdry_name += 'shiftlefttobred'
            #elif is_3_in_a_row_3d(ii, jj-1, kk, mask, 1) and is_3_in_a_row_3d(ii+1, jj-1, kk, mask, 0):
            elif is_offset_in_mask_3d(ii, jj, kk, [[1,-1,0],[2,-1,0],[1,-2,0],[0,-2,0],[0,-1,0]], mask):
              if is_offset_in_mask_3d(ii, jj, kk, [[2,-2,0],[0,-3,0],[1,-3,0]], mask):
                bdry_name += 'shiftlefttot'
              else:
                bdry_name += 'shiftlefttotred'
            #elif is_3_in_a_row_3d(ii, jj, kk+1, mask, 2) and is_3_in_a_row_3d(ii+1, jj, kk+1, mask, 0):
            elif is_offset_in_mask_3d(ii, jj, kk, [[1,0,1],[2,0,1],[1,0,2],[0,0,2],[0,0,1]], mask):
              if is_offset_in_mask_3d(ii, jj, kk, [[2,0,2],[0,0,3],[1,0,2]], mask):
                bdry_name += 'shiftlefttore'
              else:
                bdry_name += 'shiftlefttorered'
            #elif is_3_in_a_row_3d(ii, jj, kk-1, mask, 2) and is_3_in_a_row_3d(ii+1, jj, kk-1, mask, 0):
            elif is_offset_in_mask_3d(ii, jj, kk, [[1,0,-1],[2,0,-1],[1,0,-2],[0,0,-2],[0,0,-1]], mask):
              if is_offset_in_mask_3d(ii, jj, kk, [[2,0,-2],[0,0,-3],[1,0,-2]], mask):
                bdry_name += 'shiftlefttof'
              else:
                bdry_name += 'shiftlefttofred'
            #elif is_3_in_a_row_3d(ii, jj+1, kk, mask, 1) and is_3_in_a_row_3d(ii-1, jj+1, kk, mask, 0):
            elif is_offset_in_mask_3d(ii, jj, kk, [[-1,1,0],[-2,1,0],[-1,2,0],[0,2,0],[0,1,0]], mask):
              if is_offset_in_mask_3d(ii, jj, kk, [[-2,2,0],[0,3,0],[-1,3,0]], mask):
                bdry_name += 'shiftrighttob'
              else:
                bdry_name += 'shiftrighttobred'
            #elif is_3_in_a_row_3d(ii, jj-1, kk, mask, 1) and is_3_in_a_row_3d(ii-1, jj-1, kk, mask, 0):
            elif is_offset_in_mask_3d(ii, jj, kk, [[-1,-1,0],[-2,-1,0],[-1,-2,0],[0,-2,0],[0,-1,0]], mask):
              if is_offset_in_mask_3d(ii, jj, kk, [[-2,-2,0],[0,-3,0],[-1,-3,0]], mask):
                bdry_name += 'shiftrighttot'
              else:
                bdry_name += 'shiftrighttotred'
            #elif is_3_in_a_row_3d(ii, jj, kk+1, mask, 2) and is_3_in_a_row_3d(ii-1, jj, kk+1, mask, 0):
            elif is_offset_in_mask_3d(ii, jj, kk, [[-1,0,1],[-2,0,1],[-1,0,2],[0,0,2],[0,0,1]], mask):
              if is_offset_in_mask_3d(ii, jj, kk, [[-2,0,2],[0,0,3],[-1,0,3]], mask):
                bdry_name += 'shiftrighttore'
              else:
                bdry_name += 'shiftrighttorered'
            #elif is_3_in_a_row_3d(ii, jj, kk-1, mask, 2) and is_3_in_a_row_3d(ii-1, jj, kk-1, mask, 0):
            elif is_offset_in_mask_3d(ii, jj, kk, [[-1,0,-1],[-2,0,-1],[-1,0,-2],[0,0,-2],[0,0,-1]], mask):
              if is_offset_in_mask_3d(ii, jj, kk, [[-2,0,-2],[0,0,-3],[-1,0,-3]], mask):
                bdry_name += 'shiftrighttof'
              else:
                bdry_name += 'shiftrighttofred'
            else:
              bdry_name += 'nox'

            #if 'no' in bdry_name or 'shift' in bdry_name:
            if 'no' in bdry_name:
              # can't take derivatives currently
              # Note that with care there are some cases here where we still can take derivatives
              # just not as currently implemented.  For now, mask these regions
              num_no += 1
              if modify_mask:
                #print("masking pixel", ii, jj, kk,"for boundary type",bdry_name)
                mask[ii,jj,kk] = 0
              # Instead, leave points unmasked, but set boundary type to out of bounds so we don't take derivatives of these points
              # Want to be able to use the points when calculating derivatives for other points
              bdry_type[ii,jj,kk] = 0
            else:
              #if 'shift' in bdry_name:
                # ignoring all shift points
                #if modify_mask:
                  #print("ignoring shift case, for pixel", ii, jj, kk,"for boundary type",bdry_name)
                  #mask[ii,jj,kk] = 0
                #bdry_type[ii,jj,kk]=0
              if 'red' in bdry_name:
                #print("Will compute reduced accuracy 2nd derivative for pixel", ii , jj, kk, "boundary type", bdry_name)
                num_red += 1
                #if modify_mask:
                  #print("reduced 2nd deriv accuracy, for pixel", ii, jj, kk,"for boundary type",bdry_name)
                  #mask[ii,jj,kk] = 0
                #bdry_type[ii,jj,kk] = 0
              if bdry_name not in bdry_map:
                bdry_map[bdry_name] = num_known_bdrys
                num_known_bdrys += 1
              bdry_type[ii,jj,kk] = bdry_map[bdry_name]  
        
        else:
          bdry_type[ii,jj,kk] = 0


  # for ii in range(xsz):
  #   for jj in range(ysz):
  #     for kk in range(zsz):
  #       if mask[ii,jj,kk]:
  #         if is_interior_pt_3d(ii, jj, kk, mask):
  #           # interior point
  #           bdry_type[ii,jj,kk] = bdry_map["interior"]
        
  #         elif (mask[ii-1, jj, kk] and mask[ii+1, jj, kk]
  #               and mask[ii, jj-1, kk] and mask[ii, jj+1, kk]
  #               and mask[ii, jj, kk-1] and mask[ii, jj, kk+1]):
  #           # mostly interior, especially for first deriv purposes.
  #           # counts only as boundary for second derivative cross term calculations d2/dxy etc.
  #           if (mask[ii+1,jj-1,kk] and mask[ii+1,jj+1,kk]
  #               and mask[ii+1,jj,kk-1] and mask[ii+1,jj,kk+1]):
  #             # interior left
  #             bdry_type[ii,jj,kk] = bdry_map["interiorleft"]
        
  #           elif (mask[ii-1,jj-1,kk] and mask[ii-1,jj+1,kk]
  #                 and mask[ii-1,jj,kk-1] and mask[ii-1,jj,kk+1]):
  #             # interior right
  #             bdry_type[ii,jj,kk] = bdry_map["interiorright"]
        
  #           elif (mask[ii-1,jj+1,kk] and mask[ii+1,jj+1,kk]
  #                 and mask[ii,jj+1,kk-1] and mask[ii,jj+1,kk+1]):
  #             # interior bottom
  #             bdry_type[ii,jj,kk] = bdry_map["interiorbottom"]
        
  #           elif (mask[ii-1,jj-1,kk] and mask[ii+1,jj-1,kk]
  #                 and mask[ii,jj-1,kk-1] and mask[ii,jj-1,kk+1]):
  #             # interior top
  #             bdry_type[ii,jj,kk] = bdry_map["interiortop"]

  #           elif (mask[ii-1,jj,kk+1] and mask[ii+1,jj,kk+1]
  #                 and mask[ii,jj-1,kk+1] and mask[ii,jj+1,kk+1]):
  #             # interior rear
  #             bdry_type[ii,jj,kk] = bdry_map["interiorrear"]
        
  #           elif (mask[ii-1,jj,kk-1] and mask[ii+1,jj,kk-1]
  #                 and mask[ii,jj-1,kk-1] and mask[ii,jj+1,kk-1]):
  #             # interior front
  #             bdry_type[ii,jj,kk] = bdry_map["interiorfront"]
        
  #         elif is_interior_pt_3d(ii+1, jj, kk, mask):
  #           # left
  #           bdry_type[ii,jj,kk] = bdry_map["left"]
        
  #         elif is_interior_pt_3d(ii-1, jj, kk, mask):
  #           # right
  #           bdry_type[ii,jj,kk] = bdry_map["right"]
        
  #         elif is_interior_pt_3d(ii, jj+1, kk, mask):
  #           # bottom
  #           bdry_type[ii,jj,kk] = bdry_map["bottom"]
        
  #         elif is_interior_pt_3d(ii, jj-1, kk, mask):
  #           # top
  #           bdry_type[ii,jj,kk] = bdry_map["top"]

  #         elif is_interior_pt_3d(ii, jj, kk+1, mask):
  #           # bottom
  #           bdry_type[ii,jj,kk] = bdry_map["rear"]
        
  #         elif is_interior_pt_3d(ii, jj, kk-1, mask):
  #           # top
  #           bdry_type[ii,jj,kk] = bdry_map["front"]
            
  #         elif is_interior_pt_3d(ii+1, jj+1, kk, mask):
  #           # left bottom
  #           bdry_type[ii,jj,kk] = bdry_map["bottomleft"]
        
  #         elif is_interior_pt_3d(ii+1, jj-1, kk, mask):
  #           # left top
  #           bdry_type[ii,jj,kk] = bdry_map["topleft"]
        
  #         elif is_interior_pt_3d(ii-1, jj+1, kk, mask):
  #           # right bottom
  #           bdry_type[ii,jj,kk] = bdry_map["bottomright"]
        
  #         elif is_interior_pt_3d(ii-1, jj-1, kk, mask):
  #           # right top
  #           bdry_type[ii,jj,kk] = bdry_map["topright"]
            
  #         elif is_interior_pt_3d(ii+1, jj, kk+1, mask):
  #           # left rear
  #           bdry_type[ii,jj,kk] = bdry_map["rearleft"]

  #         elif is_interior_pt_3d(ii+1, jj, kk-1, mask):
  #           # left front
  #           bdry_type[ii,jj,kk] = bdry_map["frontleft"]
        
  #         elif is_interior_pt_3d(ii-1, jj, kk+1, mask):
  #           # right rear
  #           bdry_type[ii,jj,kk] = bdry_map["rearright"]
        
  #         elif is_interior_pt_3d(ii-1, jj, kk-1, mask):
  #           # right front
  #           bdry_type[ii,jj,kk] = bdry_map["frontright"]

  #         elif is_interior_pt_3d(ii, jj+1, kk+1, mask):
  #           # bottom rear
  #           bdry_type[ii,jj,kk] = bdry_map["rearbottom"]

  #         elif is_interior_pt_3d(ii, jj+1, kk-1, mask):
  #           # bottom front
  #           bdry_type[ii,jj,kk] = bdry_map["frontbottom"]
        
  #         elif is_interior_pt_3d(ii, jj-1, kk+1, mask):
  #           # top rear
  #           bdry_type[ii,jj,kk] = bdry_map["reartop"]
        
  #         elif is_interior_pt_3d(ii, jj-1, kk-1, mask):
  #           # top front
  #           bdry_type[ii,jj,kk] = bdry_map["fronttop"]
  #         elif is_interior_pt_3d(ii+1, jj+1, kk+1, mask):
  #           # left bottom rear
  #           bdry_type[ii,jj,kk] = bdry_map["rearbottomleft"]

  #         elif is_interior_pt_3d(ii+1, jj+1, kk-1, mask):
  #           # left bottom front
  #           bdry_type[ii,jj,kk] = bdry_map["frontbottomleft"]
        
  #         elif is_interior_pt_3d(ii-1, jj+1, kk+1, mask):
  #           # right bottom rear
  #           bdry_type[ii,jj,kk] = bdry_map["rearbottomright"]
        
  #         elif is_interior_pt_3d(ii-1, jj+1, kk-1, mask):
  #           # right bottom front
  #           bdry_type[ii,jj,kk] = bdry_map["frontbottomright"]

  #         elif is_interior_pt_3d(ii+1, jj-1, kk+1, mask):
  #           # left top rear
  #           bdry_type[ii,jj,kk] = bdry_map["reartopleft"]

  #         elif is_interior_pt_3d(ii+1, jj-1, kk-1, mask):
  #           # left top front
  #           bdry_type[ii,jj,kk] = bdry_map["fronttopleft"]
        
  #         elif is_interior_pt_3d(ii-1, jj-1, kk+1, mask):
  #           # right top rear
  #           bdry_type[ii,jj,kk] = bdry_map["reartopright"]
        
  #         elif is_interior_pt_3d(ii-1, jj-1, kk-1, mask):
  #           # right top front
  #           bdry_type[ii,jj,kk] = bdry_map["fronttopright"]
            
  #         elif (mask[ii-1, jj-1, kk] and mask[ii-1, jj+1, kk]
  #               and mask[ii-1, jj, kk-1] and mask[ii-1, jj, kk+1]
  #               and mask[ii-1, jj, kk] and mask[ii-2, jj, kk]):
  #           # not left
  #           # TODO decide whether better to mask or not mask these pixels
  #           # If we set mask to 0 here, then we need to set new boundary pixels accordingly!
  #           #print("notleft masking pixel", ii, jj, kk)
  #           bdry_type[ii,jj,kk] = bdry_map["notleft"]
  #           #mask[ii,jj,kk] = 0
        
  #         elif (mask[ii+1, jj-1, kk] and mask[ii+1, jj+1, kk]
  #               and mask[ii+1, jj, kk-1] and mask[ii+1, jj, kk+1]
  #               and mask[ii+1, jj, kk] and mask[ii+2, jj, kk]):
  #           # not right
  #           #print("notright masking pixel", ii, jj, kk)
  #           bdry_type[ii,jj,kk] = bdry_map["notright"]
  #           #mask[ii,jj,kk] = 0
        
  #         elif (mask[ii-1, jj-1, kk] and mask[ii+1, jj-1, kk]
  #               and mask[ii, jj-1, kk-1] and mask[ii, jj-1, kk+1]
  #               and mask[ii, jj-1, kk] and mask[ii, jj-2, kk]):
  #           # not bottom
  #           #print("notbottom masking pixel", ii, jj, kk)
  #           bdry_type[ii,jj,kk] = bdry_map["notbottom"]
  #           #mask[ii,jj,kk] = 0
        
  #         elif (mask[ii-1, jj+1, kk] and mask[ii+1, jj+1, kk]
  #               and mask[ii, jj+1, kk-1] and mask[ii, jj+1, kk+1]
  #               and mask[ii, jj+1, kk] and mask[ii, jj+2, kk]):
  #           # not top
  #           #print("nottop masking pixel", ii, jj, kk)
  #           bdry_type[ii,jj,kk] = bdry_map["nottop"]
  #           #mask[ii,jj,kk] = 0

  #         elif (mask[ii-1, jj, kk-1] and mask[ii+1, jj, kk-1]
  #               and mask[ii, jj-1, kk-1] and mask[ii, jj+1, kk-1]
  #               and mask[ii, jj, kk-1] and mask[ii, jj, kk-2]):
  #           # not rear
  #           #print("notrear masking pixel", ii, jj, kk)
  #           bdry_type[ii,jj,kk] = bdry_map["notrear"]
  #           #mask[ii,jj,kk] = 0
        
  #         elif (mask[ii-1, jj, kk+1] and mask[ii+1, jj, kk+1]
  #               and mask[ii, jj-1, kk+1] and mask[ii, jj+1, kk+1]
  #               and mask[ii, jj, kk+1] and mask[ii, jj, kk+2]):
  #           # not front
  #           #print("notfront masking pixel", ii, jj, kk)
  #           bdry_type[ii,jj,kk] = bdry_map["notfront"]
  #           #mask[ii,jj,kk] = 0
        
  #         else:
  #           # can't take derivatives currently
  #           # Note that with care there are some cases here where we still can take derivatives
  #           # just not as currently implemented.  For now, mask these regions
  #           print("masking pixel", ii, jj, kk)
  #           mask[ii,jj,kk] = 0
  #           bdry_type[ii,jj,kk] = 0
          
  #       else:
  #         bdry_type[ii,jj,kk] = 0

  print("Found", num_no, "voxels where unable to take 1st derivative.")
  print("Found", num_red, "reduced accuracy 2nd derivative voxels.")
  #print("WARNING! IGNORING ALL SHIFT BOUNDARY CASES!")
  for ii in range(1,num_known_bdrys):
    bdry_idx[ii] = fw.where(bdry_type == ii)

  return(bdry_type, bdry_idx, bdry_map)
# end determine_boundary_3d


def open_mask_2d(mask):
  # get rid of small components
  opened = ndimage.binary_opening(mask, structure=np.ones((2, 2)))
  eroded = ndimage.binary_erosion(mask, structure=np.ones((6, 6)))
  ndimage.binary_propagation(eroded, mask=opened, output=mask)
# end open_mask_2d

def open_mask_3d(mask):
  # get rid of small components
  fw, fw_name = get_framework(mask)
  if fw_name == 'numpy':
    opened = ndimage.binary_opening(mask, structure=np.ones((2, 2, 2)))
    eroded = ndimage.binary_erosion(mask, structure=np.ones((6, 6, 6)))
    ndimage.binary_propagation(eroded, mask=opened, output=mask)
  else:
    in_mask = mask.cpu().numpy()
    opened = ndimage.binary_opening(in_mask, structure=np.ones((2, 2, 2)))
    eroded = ndimage.binary_erosion(in_mask, structure=np.ones((6, 6, 6)))
    ndimage.binary_propagation(eroded, mask=opened, output=in_mask)
    mask = fw.from_numpy(in_mask)
    
# end open_mask_3d
