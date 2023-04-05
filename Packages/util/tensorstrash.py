# end tens_interp

# def tens_interp_torch(x, y, tensor_field):
#   tens = torch.zeros((2, 2))
#   eps11 = tensor_field[0, :, :]
#   eps12 = tensor_field[1, :, :]
#   eps22 = tensor_field[2, :, :]
#   if x == math.floor(x) and y == math.floor(y):
#     tens[0, 0] = eps11[int(x), int(y)]
#     tens[0, 1] = eps12[int(x), int(y)]
#     tens[1, 0] = eps12[int(x), int(y)]
#     tens[1, 1] = eps22[int(x), int(y)]
#   elif x == math.floor(x) and y != math.floor(y):
#     tens[0, 0] = abs(y - math.floor(y)) * eps11[int(x), math.ceil(y)] + \
#            abs(y - math.ceil(y)) * eps11[int(x), math.floor(y)]
#     tens[0, 1] = abs(y - math.floor(y)) * eps12[int(x), math.ceil(y)] + \
#            abs(y - math.ceil(y)) * eps12[int(x), math.floor(y)]
#     tens[1, 0] = abs(y - math.floor(y)) * eps12[int(x), math.ceil(y)] + \
#            abs(y - math.ceil(y)) * eps12[int(x), math.floor(y)]
#     tens[1, 1] = abs(y - math.floor(y)) * eps22[int(x), math.ceil(y)] + \
#            abs(y - math.ceil(y)) * eps22[int(x), math.floor(y)]
#   elif x != math.floor(x) and y == math.floor(y):
#     tens[0, 0] = abs(x - math.floor(x)) * eps11[math.ceil(x), int(y)] + \
#            abs(x - math.ceil(x)) * eps11[math.floor(x), int(y)]
#     tens[0, 1] = abs(x - math.floor(x)) * eps12[math.ceil(x), int(y)] + \
#            abs(x - math.ceil(x)) * eps12[math.floor(x), int(y)]
#     tens[1, 0] = abs(x - math.floor(x)) * eps12[math.ceil(x), int(y)] + \
#            abs(x - math.ceil(x)) * eps12[math.floor(x), int(y)]
#     tens[1, 1] = abs(x - math.floor(x)) * eps22[math.ceil(x), int(y)] + \
#            abs(x - math.ceil(x)) * eps22[math.floor(x), int(y)]
#   else:
#     tens[0, 0] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * eps11[math.ceil(x), math.ceil(y)] \
#            + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * eps11[math.ceil(x), math.floor(y)] \
#            + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * eps11[math.floor(x), math.ceil(y)] \
#            + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * eps11[math.floor(x), math.floor(y)]
#     tens[0, 1] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * eps12[math.ceil(x), math.ceil(y)] \
#            + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * eps12[math.ceil(x), math.floor(y)] \
#            + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * eps12[math.floor(x), math.ceil(y)] \
#            + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * eps12[math.floor(x), math.floor(y)]
#     tens[1, 0] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * eps12[math.ceil(x), math.ceil(y)] \
#            + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * eps12[math.ceil(x), math.floor(y)] \
#            + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * eps12[math.floor(x), math.ceil(y)] \
#            + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * eps12[math.floor(x), math.floor(y)]
#     tens[1, 1] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * eps22[math.ceil(x), math.ceil(y)] \
#            + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * eps22[math.ceil(x), math.floor(y)] \
#            + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * eps22[math.floor(x), math.ceil(y)] \
#            + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * eps22[math.floor(x), math.floor(y)]

#   return (tens)



# def batch_tens_interp_2d(x, y, tensor_mat):
#       #print("Warning! reenable njit")
#   num_tens = x.shape[0]
#   tens = torch.zeros((num_tens, 2, 2))
#   eps11 = tensor_mat[...,0,0]
#   eps12 = tensor_mat[...,0,1]
#   eps21 = tensor_mat[...,1,0]
#   eps22 = tensor_mat[...,1,1]

#   x = torch.where(x<torch.zeros_like(x),torch.zeros_like(x),x)
#   x = torch.where(x>=torch.ones_like(x)*(eps11.shape[0]-1),torch.ones_like(x)*(eps11.shape[0]-1),x)
#   y = torch.where(y<torch.zeros_like(y),torch.zeros_like(y),y)
#   y = torch.where(y>=torch.ones_like(x)*(eps11.shape[1]-1),torch.ones_like(x)*(eps11.shape[1]-1),y)

#   ceil_x = torch.ceil(x).int()
#   floor_x = torch.floor(x).int()
#   ceil_y = torch.ceil(y).int()
#   floor_y = torch.floor(y).int()
#   x_minus_floor_x = torch.abs(x - floor_x)
#   x_minus_ceil_x = torch.abs(x - ceil_x)
#   y_minus_floor_y = torch.abs(y - floor_y)
#   y_minus_ceil_y = torch.abs(y - ceil_y)
  
#   for p in range(num_tens):
#     if x[p] == floor_x[p] and y[p] == floor_y[p]:
#       # Find index where no interpolation is needed and just copy the values
#       tens[p,0,0] = eps11[floor_x[p],floor_y[p]]
#       tens[p,0,1] = eps12[floor_x[p],floor_y[p]]
#       tens[p,1,0] = tens[p,0,1]
#       tens[p,1,1] = eps22[floor_x[p],floor_y[p]]
#     elif x[p] != floor_x[p] and y[p] == floor_y[p]:      
#       tens[p,0,0] = x_minus_floor_x[p] * eps11[ceil_x[p], y[p].int()] \
#            + x_minus_ceil_x[p] * eps11[floor_x[p], y[p].int()] 
#       tens[p,0,1] = x_minus_floor_x[p] * eps12[ceil_x[p], y[p].int()] \
#            + x_minus_ceil_x[p] * eps12[floor_x[p], y[p].int()] 
#       tens[p,1,0] = tens[p,0,1]
#       tens[p,1,1] = x_minus_floor_x[p] * eps22[ceil_x[p], y[p].int()] \
#            + x_minus_ceil_x[p] * eps22[floor_x[p], y[p].int()] 
#     elif x[p] == floor_x[p] and y[p] != floor_y[p]:
#       tens[p,0,0] = y_minus_floor_y[p] * eps11[x[p].int(), ceil_y[p]] \
#            + y_minus_ceil_y[p] * eps11[x[p].int(), floor_y[p]]
#       tens[p,0,1] = y_minus_floor_y[p] * eps12[x[p].int(), ceil_y[p]] \
#            + y_minus_ceil_y[p] * eps12[x[p].int(), floor_y[p]]
#       tens[p,1,0] = tens[p,0,1]
#       tens[p,1,1] = y_minus_floor_y[p] * eps22[x[p].int(), ceil_y[p]] \
#            + y_minus_ceil_y[p] * eps22[x[p].int(), floor_y[p]]
#     else:
#       floor_x_floor_y = x_minus_floor_x[p] * y_minus_floor_y[p]
#       floor_x_ceil_y = x_minus_floor_x[p] * y_minus_ceil_y[p]
#       ceil_x_floor_y = x_minus_ceil_x[p] * y_minus_floor_y[p]
#       ceil_x_ceil_y = x_minus_ceil_x[p] * y_minus_ceil_y[p]
#       tens[p,0,0] = floor_x_floor_y * eps11[ceil_x[p], ceil_y[p]] \
#            + floor_x_ceil_y * eps11[ceil_x[p], floor_y[p]] \
#            + ceil_x_floor_y * eps11[floor_x[p], ceil_y[p]] \
#            + ceil_x_ceil_y * eps11[floor_x[p], floor_y[p]] 
#       tens[p,0,1] = floor_x_floor_y * eps12[ceil_x[p], ceil_y[p]] \
#            + floor_x_ceil_y * eps12[ceil_x[p], floor_y[p]] \
#            + ceil_x_floor_y * eps12[floor_x[p], ceil_y[p]] \
#            + ceil_x_ceil_y * eps12[floor_x[p], floor_y[p]] 
#       tens[p,1,0] = tens[p,0,1]
#       tens[p,1,1] = floor_x_floor_y * eps22[ceil_x[p], ceil_y[p]] \
#            + floor_x_ceil_y * eps22[ceil_x[p], floor_y[p]] \
#            + ceil_x_floor_y * eps22[floor_x[p], ceil_y[p]] \
#            + ceil_x_ceil_y * eps22[floor_x[p], floor_y[p]] 
  
#   return (tens)


# def batch_vect_interp_2d(x, y, vector_field):
#   num_tens = x.shape[0]
#   vect = torch.zeros((num_tens, 2))
#   v1 = vector_field[0]
#   v2 = vector_field[1]

#   for p in range(num_tens):
#     if ((math.floor(x[p]) < 0) or (math.ceil(x[p]) >= v1.shape[0]) or (math.floor(y[p]) < 0) or (math.ceil(y[p]) >= v1.shape[1])):
#       # data is out of bounds, return identity vector
#       vect[p,0] = 1
#       vect[p,1] = 1
#       continue

#     if x[p] == math.floor(x[p]) and y[p] == math.floor(y[p]):
#       vect[p,0] = v1[int(x[p]), int(y[p])]
#       vect[p,1] = v2[int(x[p]), int(y[p])]
#     elif x[p] != math.floor(x[p]) and y[p] == math.floor(y[p]):
#       vect[p,0] = abs(x[p] - math.floor(x[p])) * v1[math.ceil(x[p]), int(y[p])] \
#                      + abs(x[p] - math.ceil(x[p])) * v1[math.floor(x[p]), int(y[p])]
#       vect[p,1] = abs(x[p] - math.floor(x[p])) * v2[math.ceil(x[p]), int(y[p])] \
#                      + abs(x[p] - math.ceil(x[p])) * v2[math.floor(x[p]), int(y[p])]
#     elif x[p] == math.floor(x[p]) and y[p] != math.floor(y[p]):
#       vect[p,0] = abs(y[p] - math.floor(y[p])) * v1[int(x[p]), math.ceil(y[p])] \
#                      + abs(y[p] - math.ceil(y[p])) * v1[int(x[p]), math.floor(y[p])]
#       vect[p,1] = abs(y[p] - math.floor(y[p])) * v2[int(x[p]), math.ceil(y[p])] \
#                      + abs(y[p] - math.ceil(y[p])) * v2[int(x[p]), math.floor(y[p])]
#     else:
#       vect[p,0] = abs(x[p] - math.floor(x[p])) * abs(y[p] - math.floor(y[p])) * v1[math.ceil(x[p]), math.ceil(y[p])] \
#                + abs(x[p] - math.floor(x[p])) * abs(y[p] - math.ceil(y[p])) * v1[math.ceil(x[p]), math.floor(y[p])] \
#                + abs(x[p] - math.ceil(x[p])) * abs(y[p] - math.floor(y[p])) * v1[math.floor(x[p]), math.ceil(y[p])] \
#                + abs(x[p] - math.ceil(x[p])) * abs(y[p] - math.ceil(y[p])) * v1[math.floor(x[p]), math.floor(y[p])] 
#       vect[p,1] = abs(x[p] - math.floor(x[p])) * abs(y[p] - math.floor(y[p])) * v2[math.ceil(x[p]), math.ceil(y[p])] \
#                + abs(x[p] - math.floor(x[p])) * abs(y[p] - math.ceil(y[p])) * v2[math.ceil(x[p]), math.floor(y[p])] \
#                + abs(x[p] - math.ceil(x[p])) * abs(y[p] - math.floor(y[p])) * v2[math.floor(x[p]), math.ceil(y[p])] \
#                + abs(x[p] - math.ceil(x[p])) * abs(y[p] - math.ceil(y[p])) * v2[math.floor(x[p]), math.floor(y[p])] 
#   return (vect)

# def tens_interp(x, y, tensor_field):
#   tens = np.zeros((2, 2))
#   eps11 = tensor_field[0, :, :]
#   eps12 = tensor_field[1, :, :]
#   eps22 = tensor_field[2, :, :]
#   if (math.floor(x) < 0) or (math.ceil(x) >= eps11.shape[0]) or (math.floor(y) < 0) or (math.ceil(y) >= eps11.shape[1]):
#      # data is out of bounds, return identity tensor
#      tens[0,0] = 1
#      tens[1,1] = 1
#      return(tens)
#   if x == math.floor(x) and y == math.floor(y):
#     tens[0, 0] = eps11[int(x), int(y)]
#     tens[0, 1] = eps12[int(x), int(y)]
#     tens[1, 0] = eps12[int(x), int(y)]
#     tens[1, 1] = eps22[int(x), int(y)]
#   elif x == math.floor(x) and y != math.floor(y):
#     tens[0, 0] = abs(y - math.floor(y)) * eps11[int(x), math.ceil(y)] + \
#            abs(y - math.ceil(y)) * eps11[int(x), math.floor(y)]
#     tens[0, 1] = abs(y - math.floor(y)) * eps12[int(x), math.ceil(y)] + \
#            abs(y - math.ceil(y)) * eps12[int(x), math.floor(y)]
#     tens[1, 0] = abs(y - math.floor(y)) * eps12[int(x), math.ceil(y)] + \
#            abs(y - math.ceil(y)) * eps12[int(x), math.floor(y)]
#     tens[1, 1] = abs(y - math.floor(y)) * eps22[int(x), math.ceil(y)] + \
#            abs(y - math.ceil(y)) * eps22[int(x), math.floor(y)]
#   elif x != math.floor(x) and y == math.floor(y):
#     tens[0, 0] = abs(x - math.floor(x)) * eps11[math.ceil(x), int(y)] + \
#            abs(x - math.ceil(x)) * eps11[math.floor(x), int(y)]
#     tens[0, 1] = abs(x - math.floor(x)) * eps12[math.ceil(x), int(y)] + \
#            abs(x - math.ceil(x)) * eps12[math.floor(x), int(y)]
#     tens[1, 0] = abs(x - math.floor(x)) * eps12[math.ceil(x), int(y)] + \
#            abs(x - math.ceil(x)) * eps12[math.floor(x), int(y)]
#     tens[1, 1] = abs(x - math.floor(x)) * eps22[math.ceil(x), int(y)] + \
#            abs(x - math.ceil(x)) * eps22[math.floor(x), int(y)]
#   else:
#     tens[0, 0] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * eps11[math.ceil(x), math.ceil(y)] \
#            + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * eps11[math.ceil(x), math.floor(y)] \
#            + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * eps11[math.floor(x), math.ceil(y)] \
#            + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * eps11[math.floor(x), math.floor(y)]
#     tens[0, 1] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * eps12[math.ceil(x), math.ceil(y)] \
#            + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * eps12[math.ceil(x), math.floor(y)] \
#            + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * eps12[math.floor(x), math.ceil(y)] \
#            + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * eps12[math.floor(x), math.floor(y)]
#     tens[1, 0] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * eps12[math.ceil(x), math.ceil(y)] \
#            + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * eps12[math.ceil(x), math.floor(y)] \
#            + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * eps12[math.floor(x), math.ceil(y)] \
#            + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * eps12[math.floor(x), math.floor(y)]
#     tens[1, 1] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * eps22[math.ceil(x), math.ceil(y)] \
#            + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * eps22[math.ceil(x), math.floor(y)] \
#            + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * eps22[math.floor(x), math.ceil(y)] \
#            + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * eps22[math.floor(x), math.floor(y)]

#   return (tens)


# def tens_interp_3d_torch(x, y, z, tensor_field):
#   tens = torch.zeros((3, 3))
#   eps11 = tensor_field[0, :, :]
#   eps12 = tensor_field[1, :, :]
#   eps13 = tensor_field[2, :, :]
#   eps22 = tensor_field[3, :, :]
#   eps23 = tensor_field[4, :, :]
#   eps33 = tensor_field[5, :, :]
#   if ((math.floor(x) < 0) or (math.ceil(x) >= eps11.shape[0])
#       or (math.floor(y) < 0) or (math.ceil(y) >= eps11.shape[1])
#       or (math.floor(z) < 0) or (math.ceil(z) >= eps11.shape[2])):
#      # data is out of bounds, return identity tensor
#      tens[0,0] = 1
#      tens[1,1] = 1
#      tens[2,2] = 1
#      return(tens)

   
#   if x == math.floor(x) and y == math.floor(y) and z == math.floor(z):
#     tens[0, 0] = eps11[int(x), int(y), int(z)]
#     tens[0, 1] = eps12[int(x), int(y), int(z)]
#     tens[1, 0] = eps12[int(x), int(y), int(z)]
#     tens[0, 2] = eps13[int(x), int(y), int(z)]
#     tens[2, 0] = eps13[int(x), int(y), int(z)]
#     tens[1, 1] = eps22[int(x), int(y), int(z)]
#     tens[1, 2] = eps23[int(x), int(y), int(z)]
#     tens[2, 1] = eps23[int(x), int(y), int(z)]
#     tens[2, 2] = eps33[int(x), int(y), int(z)]
#   elif x == math.floor(x) and y != math.floor(y) and z != math.floor(z):
#     tens[0, 0] = abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps11[int(x), math.ceil(y), math.ceil(z)] \
#                  + abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps11[int(x), math.floor(y), math.ceil(z)] \
#                  + abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps11[int(x), math.ceil(y), math.floor(z)] \
#                  + abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps11[int(x), math.floor(y), math.floor(z)] 
#     tens[0, 1] = abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps12[int(x), math.ceil(y), math.ceil(z)] \
#                  + abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps12[int(x), math.floor(y), math.ceil(z)] \
#                  + abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps12[int(x), math.ceil(y), math.floor(z)] \
#                  + abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps12[int(x), math.floor(y), math.floor(z)] 
#     tens[1, 0] = tens[0,1]
#     tens[0, 2] = abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps13[int(x), math.ceil(y), math.ceil(z)] \
#                  + abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps13[int(x), math.floor(y), math.ceil(z)] \
#                  + abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps13[int(x), math.ceil(y), math.floor(z)] \
#                  + abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps13[int(x), math.floor(y), math.floor(z)] 
#     tens[2, 0] = tens[0,2]
#     tens[1, 1] = abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps22[int(x), math.ceil(y), math.ceil(z)] \
#                  + abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps22[int(x), math.floor(y), math.ceil(z)] \
#                  + abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps22[int(x), math.ceil(y), math.floor(z)] \
#                  + abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps22[int(x), math.floor(y), math.floor(z)] 
#     tens[1, 2] = abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps23[int(x), math.ceil(y), math.ceil(z)] \
#                  + abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps23[int(x), math.floor(y), math.ceil(z)] \
#                  + abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps23[int(x), math.ceil(y), math.floor(z)] \
#                  + abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps23[int(x), math.floor(y), math.floor(z)] 
#     tens[2, 1] = tens[1,2]
#     tens[2, 2] = abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps33[int(x), math.ceil(y), math.ceil(z)] \
#                  + abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps33[int(x), math.floor(y), math.ceil(z)] \
#                  + abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps33[int(x), math.ceil(y), math.floor(z)] \
#                  + abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps33[int(x), math.floor(y), math.floor(z)]
#   elif x == math.floor(x) and y == math.floor(y) and z != math.floor(z):
#     tens[0, 0] = abs(z - math.floor(z)) * eps11[int(x), int(y), math.ceil(z)] \
#                  + abs(z - math.ceil(z)) * eps11[int(x), int(y), math.floor(z)] 
#     tens[0, 1] = abs(z - math.floor(z)) * eps12[int(x), int(y), math.ceil(z)] \
#                  + abs(z - math.ceil(z)) * eps12[int(x), int(y), math.floor(z)] 
#     tens[1, 0] = tens[0,1]
#     tens[0, 2] = abs(z - math.floor(z)) * eps13[int(x), int(y), math.ceil(z)] \
#                  + abs(z - math.ceil(z)) * eps13[int(x), int(y), math.floor(z)] 
#     tens[2, 0] = tens[0,2]
#     tens[1, 1] = abs(z - math.floor(z)) * eps22[int(x), int(y), math.ceil(z)] \
#                  + abs(z - math.ceil(z)) * eps22[int(x), int(y), math.floor(z)] 
#     tens[1, 2] = abs(z - math.floor(z)) * eps23[int(x), int(y), math.ceil(z)] \
#                  + abs(z - math.ceil(z)) * eps23[int(x), int(y), math.floor(z)] 
#     tens[2, 1] = tens[1,2]
#     tens[2, 2] = abs(z - math.floor(z)) * eps33[int(x), int(y), math.ceil(z)] \
#                  + abs(z - math.ceil(z)) * eps33[int(x), int(y), math.floor(z)]   
#   elif x != math.floor(x) and y == math.floor(y) and z != math.floor(z):
#     tens[0, 0] = abs(x - math.floor(x)) * abs(z - math.floor(z)) * eps11[math.ceil(x), int(y), math.ceil(z)] \
#                  + abs(x - math.ceil(x)) * abs(z - math.floor(z)) * eps11[math.floor(x), int(y), math.ceil(z)] \
#                  + abs(x - math.floor(x)) * abs(z - math.ceil(z)) * eps11[math.ceil(x), int(y), math.floor(z)] \
#                  + abs(x - math.ceil(x)) * abs(z - math.ceil(z)) * eps11[math.floor(x), int(y), math.floor(z)]
#     tens[0, 1] = abs(x - math.floor(x)) * abs(z - math.floor(z)) * eps12[math.ceil(x), int(y), math.ceil(z)] \
#                  + abs(x - math.ceil(x)) * abs(z - math.floor(z)) * eps12[math.floor(x), int(y), math.ceil(z)] \
#                  + abs(x - math.floor(x)) * abs(z - math.ceil(z)) * eps12[math.ceil(x), int(y), math.floor(z)] \
#                  + abs(x - math.ceil(x)) * abs(z - math.ceil(z)) * eps12[math.floor(x), int(y), math.floor(z)]
#     tens[1, 0] = tens[0,1]
#     tens[0, 2] = abs(x - math.floor(x)) * abs(z - math.floor(z)) * eps13[math.ceil(x), int(y), math.ceil(z)] \
#                  + abs(x - math.ceil(x)) * abs(z - math.floor(z)) * eps13[math.floor(x), int(y), math.ceil(z)] \
#                  + abs(x - math.floor(x)) * abs(z - math.ceil(z)) * eps13[math.ceil(x), int(y), math.floor(z)] \
#                  + abs(x - math.ceil(x)) * abs(z - math.ceil(z)) * eps13[math.floor(x), int(y), math.floor(z)]
#     tens[2, 0] = tens[0,2]
#     tens[1, 1] = abs(x - math.floor(x)) * abs(z - math.floor(z)) * eps22[math.ceil(x), int(y), math.ceil(z)] \
#                  + abs(x - math.ceil(x)) * abs(z - math.floor(z)) * eps22[math.floor(x), int(y), math.ceil(z)] \
#                  + abs(x - math.floor(x)) * abs(z - math.ceil(z)) * eps22[math.ceil(x), int(y), math.floor(z)] \
#                  + abs(x - math.ceil(x)) * abs(z - math.ceil(z)) * eps22[math.floor(x), int(y), math.floor(z)]
#     tens[1, 2] = abs(x - math.floor(x)) * abs(z - math.floor(z)) * eps23[math.ceil(x), int(y), math.ceil(z)] \
#                  + abs(x - math.ceil(x)) * abs(z - math.floor(z)) * eps23[math.floor(x), int(y), math.ceil(z)] \
#                  + abs(x - math.floor(x)) * abs(z - math.ceil(z)) * eps23[math.ceil(x), int(y), math.floor(z)] \
#                  + abs(x - math.ceil(x)) * abs(z - math.ceil(z)) * eps23[math.floor(x), int(y), math.floor(z)]
#     tens[2, 1] = tens[1,2]
#     tens[2, 2] = abs(x - math.floor(x)) * abs(z - math.floor(z)) * eps33[math.ceil(x), int(y), math.ceil(z)] \
#                  + abs(x - math.ceil(x)) * abs(z - math.floor(z)) * eps33[math.floor(x), int(y), math.ceil(z)] \
#                  + abs(x - math.floor(x)) * abs(z - math.ceil(z)) * eps33[math.ceil(x), int(y), math.floor(z)] \
#                  + abs(x - math.ceil(x)) * abs(z - math.ceil(z)) * eps33[math.floor(x), int(y), math.floor(z)]
#   elif x != math.floor(x) and y == math.floor(y) and z == math.floor(z):
#     tens[0, 0] = abs(x - math.floor(x)) * eps11[math.ceil(x), int(y),int(z)] \
#                  + abs(x - math.ceil(x)) * eps11[math.floor(x), int(y), int(z)]
#     tens[0, 1] = abs(x - math.floor(x)) * eps12[math.ceil(x), int(y),int(z)] \
#                  + abs(x - math.ceil(x)) * eps12[math.floor(x), int(y), int(z)]
#     tens[1, 0] = tens[0,1]
#     tens[0, 2] = abs(x - math.floor(x)) * eps13[math.ceil(x), int(y),int(z)] \
#                  + abs(x - math.ceil(x)) * eps13[math.floor(x), int(y), int(z)]
#     tens[2, 0] = tens[0,2]
#     tens[1, 1] = abs(x - math.floor(x)) * eps22[math.ceil(x), int(y),int(z)] \
#                  + abs(x - math.ceil(x)) * eps22[math.floor(x), int(y), int(z)]
#     tens[1, 2] = abs(x - math.floor(x)) * eps23[math.ceil(x), int(y),int(z)] \
#                  + abs(x - math.ceil(x)) * eps23[math.floor(x), int(y), int(z)]
#     tens[2, 1] = tens[1,2]
#     tens[2, 2] = abs(x - math.floor(x)) * eps33[math.ceil(x), int(y),int(z)] \
#                  + abs(x - math.ceil(x)) * eps33[math.floor(x), int(y), int(z)]
#   elif x != math.floor(x) and y != math.floor(y) and z == math.floor(z):
#     tens[0, 0] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * eps11[math.ceil(x), math.ceil(y), int(z)] \
#                  + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * eps11[math.floor(x), math.ceil(y), int(z)] \
#                  + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * eps11[math.ceil(x), math.floor(y), int(z)] \
#                  + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * eps11[math.floor(x), math.floor(y), int(z)]  
#     tens[0, 1] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * eps12[math.ceil(x), math.ceil(y), int(z)] \
#                  + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * eps12[math.floor(x), math.ceil(y), int(z)] \
#                  + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * eps12[math.ceil(x), math.floor(y), int(z)] \
#                  + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * eps12[math.floor(x), math.floor(y), int(z)]  
#     tens[1, 0] = tens[0, 1]
#     tens[0, 2] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * eps13[math.ceil(x), math.ceil(y), int(z)] \
#                  + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * eps13[math.floor(x), math.ceil(y), int(z)] \
#                  + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * eps13[math.ceil(x), math.floor(y), int(z)] \
#                  + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * eps13[math.floor(x), math.floor(y), int(z)]  
#     tens[2, 0] = tens[0, 2]
#     tens[1, 1] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * eps22[math.ceil(x), math.ceil(y), int(z)] \
#                  + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * eps22[math.floor(x), math.ceil(y), int(z)] \
#                  + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * eps22[math.ceil(x), math.floor(y), int(z)] \
#                  + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * eps22[math.floor(x), math.floor(y), int(z)]  
#     tens[1, 2] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * eps23[math.ceil(x), math.ceil(y), int(z)] \
#                  + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * eps23[math.floor(x), math.ceil(y), int(z)] \
#                  + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * eps23[math.ceil(x), math.floor(y), int(z)] \
#                  + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * eps23[math.floor(x), math.floor(y), int(z)]  
#     tens[2, 1] = tens[1, 2]
#     tens[2, 2] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * eps33[math.ceil(x), math.ceil(y), int(z)] \
#                  + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * eps33[math.floor(x), math.ceil(y), int(z)] \
#                  + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * eps33[math.ceil(x), math.floor(y), int(z)] \
#                  + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * eps33[math.floor(x), math.floor(y), int(z)]  
#   elif x == math.floor(x) and y != math.floor(y) and z == math.floor(z):
#     tens[0, 0] = abs(y - math.floor(y)) * eps11[int(x), math.ceil(y), int(z)] \
#                  + abs(y - math.ceil(y)) * eps11[int(x), math.floor(y), int(z)]
#     tens[0, 1] = abs(y - math.floor(y)) * eps12[int(x), math.ceil(y), int(z)] \
#                  + abs(y - math.ceil(y)) * eps12[int(x), math.floor(y), int(z)]
#     tens[1, 0] = tens[0,1]
#     tens[0, 2] = abs(y - math.floor(y)) * eps13[int(x), math.ceil(y), int(z)] \
#                  + abs(y - math.ceil(y)) * eps13[int(x), math.floor(y), int(z)]
#     tens[2, 0] = tens[0,2]
#     tens[1, 1] = abs(y - math.floor(y)) * eps22[int(x), math.ceil(y), int(z)] \
#                  + abs(y - math.ceil(y)) * eps22[int(x), math.floor(y), int(z)]
#     tens[1, 2] = abs(y - math.floor(y)) * eps23[int(x), math.ceil(y), int(z)] \
#                  + abs(y - math.ceil(y)) * eps23[int(x), math.floor(y), int(z)]
#     tens[2, 1] = tens[1,2]
#     tens[2, 2] = abs(y - math.floor(y)) * eps33[int(x), math.ceil(y), int(z)] \
#                  + abs(y - math.ceil(y)) * eps33[int(x), math.floor(y), int(z)]
#   else:
#     tens[0, 0] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps11[math.ceil(x), math.ceil(y), math.ceil(z)] \
#            + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps11[math.ceil(x), math.floor(y), math.ceil(z)] \
#            + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps11[math.floor(x), math.ceil(y), math.ceil(z)] \
#            + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps11[math.floor(x), math.floor(y), math.ceil(z)] \
#            + abs(x - math.floor(x)) * abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps11[math.ceil(x), math.ceil(y), math.floor(z)] \
#            + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps11[math.ceil(x), math.floor(y), math.floor(z)] \
#            + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps11[math.floor(x), math.ceil(y), math.floor(z)] \
#            + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps11[math.floor(x), math.floor(y), math.floor(z)]
#     tens[0, 1] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps12[math.ceil(x), math.ceil(y), math.ceil(z)] \
#            + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps12[math.ceil(x), math.floor(y), math.ceil(z)] \
#            + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps12[math.floor(x), math.ceil(y), math.ceil(z)] \
#            + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps12[math.floor(x), math.floor(y), math.ceil(z)] \
#            + abs(x - math.floor(x)) * abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps12[math.ceil(x), math.ceil(y), math.floor(z)] \
#            + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps12[math.ceil(x), math.floor(y), math.floor(z)] \
#            + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps12[math.floor(x), math.ceil(y), math.floor(z)] \
#            + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps12[math.floor(x), math.floor(y), math.floor(z)]
#     tens[1, 0] = tens[0,1]
#     tens[0, 2] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps13[math.ceil(x), math.ceil(y), math.ceil(z)] \
#            + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps13[math.ceil(x), math.floor(y), math.ceil(z)] \
#            + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps13[math.floor(x), math.ceil(y), math.ceil(z)] \
#            + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps13[math.floor(x), math.floor(y), math.ceil(z)] \
#            + abs(x - math.floor(x)) * abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps13[math.ceil(x), math.ceil(y), math.floor(z)] \
#            + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps13[math.ceil(x), math.floor(y), math.floor(z)] \
#            + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps13[math.floor(x), math.ceil(y), math.floor(z)] \
#            + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps13[math.floor(x), math.floor(y), math.floor(z)]
#     tens[2, 0] = tens[0,2]
#     tens[1, 1] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps22[math.ceil(x), math.ceil(y), math.ceil(z)] \
#            + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps22[math.ceil(x), math.floor(y), math.ceil(z)] \
#            + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps22[math.floor(x), math.ceil(y), math.ceil(z)] \
#            + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps22[math.floor(x), math.floor(y), math.ceil(z)] \
#            + abs(x - math.floor(x)) * abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps22[math.ceil(x), math.ceil(y), math.floor(z)] \
#            + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps22[math.ceil(x), math.floor(y), math.floor(z)] \
#            + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps22[math.floor(x), math.ceil(y), math.floor(z)] \
#            + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps22[math.floor(x), math.floor(y), math.floor(z)]
#     tens[1, 2] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps23[math.ceil(x), math.ceil(y), math.ceil(z)] \
#            + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps23[math.ceil(x), math.floor(y), math.ceil(z)] \
#            + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps23[math.floor(x), math.ceil(y), math.ceil(z)] \
#            + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps23[math.floor(x), math.floor(y), math.ceil(z)] \
#            + abs(x - math.floor(x)) * abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps23[math.ceil(x), math.ceil(y), math.floor(z)] \
#            + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps23[math.ceil(x), math.floor(y), math.floor(z)] \
#            + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps23[math.floor(x), math.ceil(y), math.floor(z)] \
#            + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps23[math.floor(x), math.floor(y), math.floor(z)]
#     tens[2,1] = tens[1,2]
#     tens[2, 2] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps33[math.ceil(x), math.ceil(y), math.ceil(z)] \
#            + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps33[math.ceil(x), math.floor(y), math.ceil(z)] \
#            + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps33[math.floor(x), math.ceil(y), math.ceil(z)] \
#            + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps33[math.floor(x), math.floor(y), math.ceil(z)] \
#            + abs(x - math.floor(x)) * abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps33[math.ceil(x), math.ceil(y), math.floor(z)] \
#            + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps33[math.ceil(x), math.floor(y), math.floor(z)] \
#            + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps33[math.floor(x), math.ceil(y), math.floor(z)] \
#            + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps33[math.floor(x), math.floor(y), math.floor(z)]

#   return (tens)

def tens_interp_3d(x, y, z, tensor_field):
  tens = np.zeros((3, 3))
  eps11 = tensor_field[0, :, :]
  eps12 = tensor_field[1, :, :]
  eps13 = tensor_field[2, :, :]
  eps22 = tensor_field[3, :, :]
  eps23 = tensor_field[4, :, :]
  eps33 = tensor_field[5, :, :]
  if ((math.floor(x) < 0) or (math.ceil(x) >= eps11.shape[0])
      or (math.floor(y) < 0) or (math.ceil(y) >= eps11.shape[1])
      or (math.floor(z) < 0) or (math.ceil(z) >= eps11.shape[2])):
     # data is out of bounds, return identity tensor
     tens[0,0] = 1
     tens[1,1] = 1
     tens[2,2] = 1
     return(tens)

   
  if x == math.floor(x) and y == math.floor(y) and z == math.floor(z):
    tens[0, 0] = eps11[int(x), int(y), int(z)]
    tens[0, 1] = eps12[int(x), int(y), int(z)]
    tens[1, 0] = eps12[int(x), int(y), int(z)]
    tens[0, 2] = eps13[int(x), int(y), int(z)]
    tens[2, 0] = eps13[int(x), int(y), int(z)]
    tens[1, 1] = eps22[int(x), int(y), int(z)]
    tens[1, 2] = eps23[int(x), int(y), int(z)]
    tens[2, 1] = eps23[int(x), int(y), int(z)]
    tens[2, 2] = eps33[int(x), int(y), int(z)]
  elif x == math.floor(x) and y != math.floor(y) and z != math.floor(z):
    tens[0, 0] = abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps11[int(x), math.ceil(y), math.ceil(z)] \
                 + abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps11[int(x), math.floor(y), math.ceil(z)] \
                 + abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps11[int(x), math.ceil(y), math.floor(z)] \
                 + abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps11[int(x), math.floor(y), math.floor(z)] 
    tens[0, 1] = abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps12[int(x), math.ceil(y), math.ceil(z)] \
                 + abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps12[int(x), math.floor(y), math.ceil(z)] \
                 + abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps12[int(x), math.ceil(y), math.floor(z)] \
                 + abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps12[int(x), math.floor(y), math.floor(z)] 
    tens[1, 0] = tens[0,1]
    tens[0, 2] = abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps13[int(x), math.ceil(y), math.ceil(z)] \
                 + abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps13[int(x), math.floor(y), math.ceil(z)] \
                 + abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps13[int(x), math.ceil(y), math.floor(z)] \
                 + abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps13[int(x), math.floor(y), math.floor(z)] 
    tens[2, 0] = tens[0,2]
    tens[1, 1] = abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps22[int(x), math.ceil(y), math.ceil(z)] \
                 + abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps22[int(x), math.floor(y), math.ceil(z)] \
                 + abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps22[int(x), math.ceil(y), math.floor(z)] \
                 + abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps22[int(x), math.floor(y), math.floor(z)] 
    tens[1, 2] = abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps23[int(x), math.ceil(y), math.ceil(z)] \
                 + abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps23[int(x), math.floor(y), math.ceil(z)] \
                 + abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps23[int(x), math.ceil(y), math.floor(z)] \
                 + abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps23[int(x), math.floor(y), math.floor(z)] 
    tens[2, 1] = tens[1,2]
    tens[2, 2] = abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps33[int(x), math.ceil(y), math.ceil(z)] \
                 + abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps33[int(x), math.floor(y), math.ceil(z)] \
                 + abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps33[int(x), math.ceil(y), math.floor(z)] \
                 + abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps33[int(x), math.floor(y), math.floor(z)]
  elif x == math.floor(x) and y == math.floor(y) and z != math.floor(z):
    tens[0, 0] = abs(z - math.floor(z)) * eps11[int(x), int(y), math.ceil(z)] \
                 + abs(z - math.ceil(z)) * eps11[int(x), int(y), math.floor(z)] 
    tens[0, 1] = abs(z - math.floor(z)) * eps12[int(x), int(y), math.ceil(z)] \
                 + abs(z - math.ceil(z)) * eps12[int(x), int(y), math.floor(z)] 
    tens[1, 0] = tens[0,1]
    tens[0, 2] = abs(z - math.floor(z)) * eps13[int(x), int(y), math.ceil(z)] \
                 + abs(z - math.ceil(z)) * eps13[int(x), int(y), math.floor(z)] 
    tens[2, 0] = tens[0,2]
    tens[1, 1] = abs(z - math.floor(z)) * eps22[int(x), int(y), math.ceil(z)] \
                 + abs(z - math.ceil(z)) * eps22[int(x), int(y), math.floor(z)] 
    tens[1, 2] = abs(z - math.floor(z)) * eps23[int(x), int(y), math.ceil(z)] \
                 + abs(z - math.ceil(z)) * eps23[int(x), int(y), math.floor(z)] 
    tens[2, 1] = tens[1,2]
    tens[2, 2] = abs(z - math.floor(z)) * eps33[int(x), int(y), math.ceil(z)] \
                 + abs(z - math.ceil(z)) * eps33[int(x), int(y), math.floor(z)]   
  elif x != math.floor(x) and y == math.floor(y) and z != math.floor(z):
    tens[0, 0] = abs(x - math.floor(x)) * abs(z - math.floor(z)) * eps11[math.ceil(x), int(y), math.ceil(z)] \
                 + abs(x - math.ceil(x)) * abs(z - math.floor(z)) * eps11[math.floor(x), int(y), math.ceil(z)] \
                 + abs(x - math.floor(x)) * abs(z - math.ceil(z)) * eps11[math.ceil(x), int(y), math.floor(z)] \
                 + abs(x - math.ceil(x)) * abs(z - math.ceil(z)) * eps11[math.floor(x), int(y), math.floor(z)]
    tens[0, 1] = abs(x - math.floor(x)) * abs(z - math.floor(z)) * eps12[math.ceil(x), int(y), math.ceil(z)] \
                 + abs(x - math.ceil(x)) * abs(z - math.floor(z)) * eps12[math.floor(x), int(y), math.ceil(z)] \
                 + abs(x - math.floor(x)) * abs(z - math.ceil(z)) * eps12[math.ceil(x), int(y), math.floor(z)] \
                 + abs(x - math.ceil(x)) * abs(z - math.ceil(z)) * eps12[math.floor(x), int(y), math.floor(z)]
    tens[1, 0] = tens[0,1]
    tens[0, 2] = abs(x - math.floor(x)) * abs(z - math.floor(z)) * eps13[math.ceil(x), int(y), math.ceil(z)] \
                 + abs(x - math.ceil(x)) * abs(z - math.floor(z)) * eps13[math.floor(x), int(y), math.ceil(z)] \
                 + abs(x - math.floor(x)) * abs(z - math.ceil(z)) * eps13[math.ceil(x), int(y), math.floor(z)] \
                 + abs(x - math.ceil(x)) * abs(z - math.ceil(z)) * eps13[math.floor(x), int(y), math.floor(z)]
    tens[2, 0] = tens[0,2]
    tens[1, 1] = abs(x - math.floor(x)) * abs(z - math.floor(z)) * eps22[math.ceil(x), int(y), math.ceil(z)] \
                 + abs(x - math.ceil(x)) * abs(z - math.floor(z)) * eps22[math.floor(x), int(y), math.ceil(z)] \
                 + abs(x - math.floor(x)) * abs(z - math.ceil(z)) * eps22[math.ceil(x), int(y), math.floor(z)] \
                 + abs(x - math.ceil(x)) * abs(z - math.ceil(z)) * eps22[math.floor(x), int(y), math.floor(z)]
    tens[1, 2] = abs(x - math.floor(x)) * abs(z - math.floor(z)) * eps23[math.ceil(x), int(y), math.ceil(z)] \
                 + abs(x - math.ceil(x)) * abs(z - math.floor(z)) * eps23[math.floor(x), int(y), math.ceil(z)] \
                 + abs(x - math.floor(x)) * abs(z - math.ceil(z)) * eps23[math.ceil(x), int(y), math.floor(z)] \
                 + abs(x - math.ceil(x)) * abs(z - math.ceil(z)) * eps23[math.floor(x), int(y), math.floor(z)]
    tens[2, 1] = tens[1,2]
    tens[2, 2] = abs(x - math.floor(x)) * abs(z - math.floor(z)) * eps33[math.ceil(x), int(y), math.ceil(z)] \
                 + abs(x - math.ceil(x)) * abs(z - math.floor(z)) * eps33[math.floor(x), int(y), math.ceil(z)] \
                 + abs(x - math.floor(x)) * abs(z - math.ceil(z)) * eps33[math.ceil(x), int(y), math.floor(z)] \
                 + abs(x - math.ceil(x)) * abs(z - math.ceil(z)) * eps33[math.floor(x), int(y), math.floor(z)]
  elif x != math.floor(x) and y == math.floor(y) and z == math.floor(z):
    tens[0, 0] = abs(x - math.floor(x)) * eps11[math.ceil(x), int(y),int(z)] \
                 + abs(x - math.ceil(x)) * eps11[math.floor(x), int(y), int(z)]
    tens[0, 1] = abs(x - math.floor(x)) * eps12[math.ceil(x), int(y),int(z)] \
                 + abs(x - math.ceil(x)) * eps12[math.floor(x), int(y), int(z)]
    tens[1, 0] = tens[0,1]
    tens[0, 2] = abs(x - math.floor(x)) * eps13[math.ceil(x), int(y),int(z)] \
                 + abs(x - math.ceil(x)) * eps13[math.floor(x), int(y), int(z)]
    tens[2, 0] = tens[0,2]
    tens[1, 1] = abs(x - math.floor(x)) * eps22[math.ceil(x), int(y),int(z)] \
                 + abs(x - math.ceil(x)) * eps22[math.floor(x), int(y), int(z)]
    tens[1, 2] = abs(x - math.floor(x)) * eps23[math.ceil(x), int(y),int(z)] \
                 + abs(x - math.ceil(x)) * eps23[math.floor(x), int(y), int(z)]
    tens[2, 1] = tens[1,2]
    tens[2, 2] = abs(x - math.floor(x)) * eps33[math.ceil(x), int(y),int(z)] \
                 + abs(x - math.ceil(x)) * eps33[math.floor(x), int(y), int(z)]
  elif x != math.floor(x) and y != math.floor(y) and z == math.floor(z):
    tens[0, 0] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * eps11[math.ceil(x), math.ceil(y), int(z)] \
                 + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * eps11[math.floor(x), math.ceil(y), int(z)] \
                 + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * eps11[math.ceil(x), math.floor(y), int(z)] \
                 + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * eps11[math.floor(x), math.floor(y), int(z)]  
    tens[0, 1] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * eps12[math.ceil(x), math.ceil(y), int(z)] \
                 + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * eps12[math.floor(x), math.ceil(y), int(z)] \
                 + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * eps12[math.ceil(x), math.floor(y), int(z)] \
                 + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * eps12[math.floor(x), math.floor(y), int(z)]  
    tens[1, 0] = tens[0, 1]
    tens[0, 2] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * eps13[math.ceil(x), math.ceil(y), int(z)] \
                 + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * eps13[math.floor(x), math.ceil(y), int(z)] \
                 + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * eps13[math.ceil(x), math.floor(y), int(z)] \
                 + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * eps13[math.floor(x), math.floor(y), int(z)]  
    tens[2, 0] = tens[0, 2]
    tens[1, 1] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * eps22[math.ceil(x), math.ceil(y), int(z)] \
                 + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * eps22[math.floor(x), math.ceil(y), int(z)] \
                 + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * eps22[math.ceil(x), math.floor(y), int(z)] \
                 + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * eps22[math.floor(x), math.floor(y), int(z)]  
    tens[1, 2] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * eps23[math.ceil(x), math.ceil(y), int(z)] \
                 + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * eps23[math.floor(x), math.ceil(y), int(z)] \
                 + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * eps23[math.ceil(x), math.floor(y), int(z)] \
                 + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * eps23[math.floor(x), math.floor(y), int(z)]  
    tens[2, 1] = tens[1, 2]
    tens[2, 2] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * eps33[math.ceil(x), math.ceil(y), int(z)] \
                 + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * eps33[math.floor(x), math.ceil(y), int(z)] \
                 + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * eps33[math.ceil(x), math.floor(y), int(z)] \
                 + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * eps33[math.floor(x), math.floor(y), int(z)]  
  elif x == math.floor(x) and y != math.floor(y) and z == math.floor(z):
    tens[0, 0] = abs(y - math.floor(y)) * eps11[int(x), math.ceil(y), int(z)] \
                 + abs(y - math.ceil(y)) * eps11[int(x), math.floor(y), int(z)]
    tens[0, 1] = abs(y - math.floor(y)) * eps12[int(x), math.ceil(y), int(z)] \
                 + abs(y - math.ceil(y)) * eps12[int(x), math.floor(y), int(z)]
    tens[1, 0] = tens[0,1]
    tens[0, 2] = abs(y - math.floor(y)) * eps13[int(x), math.ceil(y), int(z)] \
                 + abs(y - math.ceil(y)) * eps13[int(x), math.floor(y), int(z)]
    tens[2, 0] = tens[0,2]
    tens[1, 1] = abs(y - math.floor(y)) * eps22[int(x), math.ceil(y), int(z)] \
                 + abs(y - math.ceil(y)) * eps22[int(x), math.floor(y), int(z)]
    tens[1, 2] = abs(y - math.floor(y)) * eps23[int(x), math.ceil(y), int(z)] \
                 + abs(y - math.ceil(y)) * eps23[int(x), math.floor(y), int(z)]
    tens[2, 1] = tens[1,2]
    tens[2, 2] = abs(y - math.floor(y)) * eps33[int(x), math.ceil(y), int(z)] \
                 + abs(y - math.ceil(y)) * eps33[int(x), math.floor(y), int(z)]
  else:
    tens[0, 0] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps11[math.ceil(x), math.ceil(y), math.ceil(z)] \
           + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps11[math.ceil(x), math.floor(y), math.ceil(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps11[math.floor(x), math.ceil(y), math.ceil(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps11[math.floor(x), math.floor(y), math.ceil(z)] \
           + abs(x - math.floor(x)) * abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps11[math.ceil(x), math.ceil(y), math.floor(z)] \
           + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps11[math.ceil(x), math.floor(y), math.floor(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps11[math.floor(x), math.ceil(y), math.floor(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps11[math.floor(x), math.floor(y), math.floor(z)]
    tens[0, 1] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps12[math.ceil(x), math.ceil(y), math.ceil(z)] \
           + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps12[math.ceil(x), math.floor(y), math.ceil(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps12[math.floor(x), math.ceil(y), math.ceil(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps12[math.floor(x), math.floor(y), math.ceil(z)] \
           + abs(x - math.floor(x)) * abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps12[math.ceil(x), math.ceil(y), math.floor(z)] \
           + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps12[math.ceil(x), math.floor(y), math.floor(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps12[math.floor(x), math.ceil(y), math.floor(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps12[math.floor(x), math.floor(y), math.floor(z)]
    tens[1, 0] = tens[0,1]
    tens[0, 2] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps13[math.ceil(x), math.ceil(y), math.ceil(z)] \
           + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps13[math.ceil(x), math.floor(y), math.ceil(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps13[math.floor(x), math.ceil(y), math.ceil(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps13[math.floor(x), math.floor(y), math.ceil(z)] \
           + abs(x - math.floor(x)) * abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps13[math.ceil(x), math.ceil(y), math.floor(z)] \
           + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps13[math.ceil(x), math.floor(y), math.floor(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps13[math.floor(x), math.ceil(y), math.floor(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps13[math.floor(x), math.floor(y), math.floor(z)]
    tens[2, 0] = tens[0,2]
    tens[1, 1] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps22[math.ceil(x), math.ceil(y), math.ceil(z)] \
           + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps22[math.ceil(x), math.floor(y), math.ceil(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps22[math.floor(x), math.ceil(y), math.ceil(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps22[math.floor(x), math.floor(y), math.ceil(z)] \
           + abs(x - math.floor(x)) * abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps22[math.ceil(x), math.ceil(y), math.floor(z)] \
           + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps22[math.ceil(x), math.floor(y), math.floor(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps22[math.floor(x), math.ceil(y), math.floor(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps22[math.floor(x), math.floor(y), math.floor(z)]
    tens[1, 2] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps23[math.ceil(x), math.ceil(y), math.ceil(z)] \
           + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps23[math.ceil(x), math.floor(y), math.ceil(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps23[math.floor(x), math.ceil(y), math.ceil(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps23[math.floor(x), math.floor(y), math.ceil(z)] \
           + abs(x - math.floor(x)) * abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps23[math.ceil(x), math.ceil(y), math.floor(z)] \
           + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps23[math.ceil(x), math.floor(y), math.floor(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps23[math.floor(x), math.ceil(y), math.floor(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps23[math.floor(x), math.floor(y), math.floor(z)]
    tens[2,1] = tens[1,2]
    tens[2, 2] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps33[math.ceil(x), math.ceil(y), math.ceil(z)] \
           + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps33[math.ceil(x), math.floor(y), math.ceil(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * abs(z - math.floor(z)) * eps33[math.floor(x), math.ceil(y), math.ceil(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * abs(z - math.floor(z)) * eps33[math.floor(x), math.floor(y), math.ceil(z)] \
           + abs(x - math.floor(x)) * abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps33[math.ceil(x), math.ceil(y), math.floor(z)] \
           + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps33[math.ceil(x), math.floor(y), math.floor(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * abs(z - math.ceil(z)) * eps33[math.floor(x), math.ceil(y), math.floor(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * eps33[math.floor(x), math.floor(y), math.floor(z)]

  return (tens)

@njit()
def batch_tens_interp_3d(x, y, z, tensor_field):
  #print("Warning! reenable njit")
  num_tens = x.shape[0]
  tens = np.zeros((num_tens, 3, 3),dtype=np.float_)
  eps11 = tensor_field[0, :, :, :]
  eps12 = tensor_field[1, :, :, :]
  eps13 = tensor_field[2, :, :, :]
  eps22 = tensor_field[3, :, :, :]
  eps23 = tensor_field[4, :, :, :]
  eps33 = tensor_field[5, :, :, :]

  x = np.where(x<0,0,x)
  x = np.where(x>=eps11.shape[0]-1,eps11.shape[0]-1,x)
  y = np.where(y<0,0,y)
  y = np.where(y>=eps11.shape[1]-1,eps11.shape[1]-1,y)
  z = np.where(z<0,0,z)
  z = np.where(z>=eps11.shape[2]-1,eps11.shape[2]-1,z)

  ceil_x = np.ceil(x).astype(np.int_)
  floor_x = np.floor(x).astype(np.int_)
  ceil_y = np.ceil(y).astype(np.int_)
  floor_y = np.floor(y).astype(np.int_)
  ceil_z = np.ceil(z).astype(np.int_)
  floor_z = np.floor(z).astype(np.int_)
  x_minus_floor_x = np.abs(x - floor_x)
  x_minus_ceil_x = np.abs(x - ceil_x)
  y_minus_floor_y = np.abs(y - floor_y)
  y_minus_ceil_y = np.abs(y - ceil_y)
  z_minus_floor_z = np.abs(z - floor_z)
  z_minus_ceil_z = np.abs(z - ceil_z)
  


  # Find index where interpolation is needed and interpolate
  # This for loop is way too slow without numba.  Use above indexing if numba is not available
  #for p in prange(num_tens):
  for p in range(num_tens):
    if x[p] == floor_x[p] and y[p] == floor_y[p] and z[p] == floor_z[p]:
      # Find index where no interpolation is needed and just copy the values
      tens[p,0,0] = eps11[floor_x[p],floor_y[p],floor_z[p]]
      tens[p,0,1] = eps12[floor_x[p],floor_y[p],floor_z[p]]
      tens[p,1,0] = tens[p,0,1]
      tens[p,0,2] = eps13[floor_x[p],floor_y[p],floor_z[p]]
      tens[p,2,0] = tens[p,0,2]
      tens[p,1,1] = eps22[floor_x[p],floor_y[p],floor_z[p]]
      tens[p,1,2] = eps23[floor_x[p],floor_y[p],floor_z[p]]
      tens[p,2,1] = tens[p,1,2]
      tens[p,2,2] = eps33[floor_x[p],floor_y[p],floor_z[p]]
    elif x[p] == floor_x[p] and y[p] != floor_y[p] and z[p] != floor_z[p]:
      floor_y_floor_z = y_minus_floor_y[p] * z_minus_floor_z[p]
      floor_y_ceil_z = y_minus_floor_y[p] * z_minus_ceil_z[p]
      ceil_y_floor_z = y_minus_ceil_y[p] * z_minus_floor_z[p]
      ceil_y_ceil_z = y_minus_ceil_y[p] * z_minus_ceil_z[p]
      
      tens[p,0,0] = floor_y_floor_z * eps11[np.int64(x[p]), ceil_y[p], ceil_z[p]] \
           + ceil_y_floor_z * eps11[np.int64(x[p]), floor_y[p], ceil_z[p]] \
           + floor_y_ceil_z * eps11[np.int64(x[p]), ceil_y[p], floor_z[p]] \
           + ceil_y_ceil_z * eps11[np.int64(x[p]), floor_y[p], floor_z[p]] 
      tens[p,0,1] = floor_y_floor_z * eps12[np.int64(x[p]), ceil_y[p], ceil_z[p]] \
           + ceil_y_floor_z * eps12[np.int64(x[p]), floor_y[p], ceil_z[p]] \
           + floor_y_ceil_z * eps12[np.int64(x[p]), ceil_y[p], floor_z[p]] \
           + ceil_y_ceil_z * eps12[np.int64(x[p]), floor_y[p], floor_z[p]] 
      tens[p,1,0] = tens[p,0,1]
      tens[p,0,2] = floor_y_floor_z * eps13[np.int64(x[p]), ceil_y[p], ceil_z[p]] \
           + ceil_y_floor_z * eps13[np.int64(x[p]), floor_y[p], ceil_z[p]] \
           + floor_y_ceil_z * eps13[np.int64(x[p]), ceil_y[p], floor_z[p]] \
           + ceil_y_ceil_z * eps13[np.int64(x[p]), floor_y[p], floor_z[p]] 
      tens[p,2,0] = tens[p,0,2]
      tens[p,1,1] = floor_y_floor_z * eps22[np.int64(x[p]), ceil_y[p], ceil_z[p]] \
           + ceil_y_floor_z * eps22[np.int64(x[p]), floor_y[p], ceil_z[p]] \
           + floor_y_ceil_z * eps22[np.int64(x[p]), ceil_y[p], floor_z[p]] \
           + ceil_y_ceil_z * eps22[np.int64(x[p]), floor_y[p], floor_z[p]] 
      tens[p,1,2] = floor_y_floor_z * eps23[np.int64(x[p]), ceil_y[p], ceil_z[p]] \
           + ceil_y_floor_z * eps23[np.int64(x[p]), floor_y[p], ceil_z[p]] \
           + floor_y_ceil_z * eps23[np.int64(x[p]), ceil_y[p], floor_z[p]] \
           + ceil_y_ceil_z * eps23[np.int64(x[p]), floor_y[p], floor_z[p]] 
      tens[p,2,1] = tens[p,1,2]
      tens[p,2,2] = floor_y_floor_z * eps33[np.int64(x[p]), ceil_y[p], ceil_z[p]] \
           + ceil_y_floor_z * eps33[np.int64(x[p]), floor_y[p], ceil_z[p]] \
           + floor_y_ceil_z * eps33[np.int64(x[p]), ceil_y[p], floor_z[p]] \
           + ceil_y_ceil_z * eps33[np.int64(x[p]), floor_y[p], floor_z[p]] 
    elif x[p] == floor_x[p] and y[p] == floor_y[p] and z[p] != floor_z[p]:
      tens[p,0,0] = z_minus_floor_z[p] * eps11[np.int64(x[p]), np.int64(y[p]), ceil_z[p]] \
           + z_minus_ceil_z[p] * eps11[np.int64(x[p]), np.int64(y[p]), floor_z[p]]
      tens[p,0,1] = z_minus_floor_z[p] * eps12[np.int64(x[p]), np.int64(y[p]), ceil_z[p]] \
           + z_minus_ceil_z[p] * eps12[np.int64(x[p]), np.int64(y[p]), floor_z[p]]
      tens[p,1,0] = tens[p,0,1]
      tens[p,0,2] = z_minus_floor_z[p] * eps13[np.int64(x[p]), np.int64(y[p]), ceil_z[p]] \
           + z_minus_ceil_z[p] * eps13[np.int64(x[p]), np.int64(y[p]), floor_z[p]]
      tens[p,2,0] = tens[p,0,2]
      tens[p,1,1] = z_minus_floor_z[p] * eps22[np.int64(x[p]), np.int64(y[p]), ceil_z[p]] \
           + z_minus_ceil_z[p] * eps22[np.int64(x[p]), np.int64(y[p]), floor_z[p]]
      tens[p,1,2] = z_minus_floor_z[p] * eps23[np.int64(x[p]), np.int64(y[p]), ceil_z[p]] \
           + z_minus_ceil_z[p] * eps23[np.int64(x[p]), np.int64(y[p]), floor_z[p]]
      tens[p,2,1] = tens[p,1,2]
      tens[p,2,2] = z_minus_floor_z[p] * eps33[np.int64(x[p]), np.int64(y[p]), ceil_z[p]] \
           + z_minus_ceil_z[p] * eps33[np.int64(x[p]), np.int64(y[p]), floor_z[p]]
    elif x[p] != floor_x[p] and y[p] == floor_y[p] and z[p] != floor_z[p]:
      floor_x_floor_z = x_minus_floor_x[p] * z_minus_floor_z[p]
      floor_x_ceil_z = x_minus_floor_x[p] * z_minus_ceil_z[p]
      ceil_x_floor_z = x_minus_ceil_x[p] * z_minus_floor_z[p]
      ceil_x_ceil_z = x_minus_ceil_x[p] * z_minus_ceil_z[p]
      
      tens[p,0,0] = floor_x_floor_z * eps11[ceil_x[p], np.int64(y[p]), ceil_z[p]] \
           + ceil_x_floor_z * eps11[floor_x[p], np.int64(y[p]), ceil_z[p]] \
           + floor_x_ceil_z * eps11[ceil_x[p], np.int64(y[p]), floor_z[p]] \
           + ceil_x_ceil_z * eps11[floor_x[p], np.int64(y[p]), floor_z[p]] 
      tens[p,0,1] = floor_x_floor_z * eps12[ceil_x[p], np.int64(y[p]), ceil_z[p]] \
           + ceil_x_floor_z * eps12[floor_x[p], np.int64(y[p]), ceil_z[p]] \
           + floor_x_ceil_z * eps12[ceil_x[p], np.int64(y[p]), floor_z[p]] \
           + ceil_x_ceil_z * eps12[floor_x[p], np.int64(y[p]), floor_z[p]] 
      tens[p,1,0] = tens[p,0,1]
      tens[p,0,2] = floor_x_floor_z * eps13[ceil_x[p], np.int64(y[p]), ceil_z[p]] \
           + ceil_x_floor_z * eps13[floor_x[p], np.int64(y[p]), ceil_z[p]] \
           + floor_x_ceil_z * eps13[ceil_x[p], np.int64(y[p]), floor_z[p]] \
           + ceil_x_ceil_z * eps13[floor_x[p], np.int64(y[p]), floor_z[p]] 
      tens[p,2,0] = tens[p,0,2]
      tens[p,1,1] = floor_x_floor_z * eps22[ceil_x[p], np.int64(y[p]), ceil_z[p]] \
           + ceil_x_floor_z * eps22[floor_x[p], np.int64(y[p]), ceil_z[p]] \
           + floor_x_ceil_z * eps22[ceil_x[p], np.int64(y[p]), floor_z[p]] \
           + ceil_x_ceil_z * eps22[floor_x[p], np.int64(y[p]), floor_z[p]] 
      tens[p,1,2] = floor_x_floor_z * eps23[ceil_x[p], np.int64(y[p]), ceil_z[p]] \
           + ceil_x_floor_z * eps23[floor_x[p], np.int64(y[p]), ceil_z[p]] \
           + floor_x_ceil_z * eps23[ceil_x[p], np.int64(y[p]), floor_z[p]] \
           + ceil_x_ceil_z * eps23[floor_x[p], np.int64(y[p]), floor_z[p]] 
      tens[p,2,1] = tens[p,1,2]
      tens[p,2,2] = floor_x_floor_z * eps33[ceil_x[p], np.int64(y[p]), ceil_z[p]] \
           + ceil_x_floor_z * eps33[floor_x[p], np.int64(y[p]), ceil_z[p]] \
           + floor_x_ceil_z * eps33[ceil_x[p], np.int64(y[p]), floor_z[p]] \
           + ceil_x_ceil_z * eps33[floor_x[p], np.int64(y[p]), floor_z[p]] 
    elif x[p] != floor_x[p] and y[p] == floor_y[p] and z[p] == floor_z[p]:
      tens[p,0,0] = x_minus_floor_x[p] * eps11[ceil_x[p], np.int64(y[p]), np.int64(z[p])] \
           + x_minus_ceil_x[p] * eps11[floor_x[p], np.int64(y[p]), np.int64(z[p])]
      tens[p,0,1] = x_minus_floor_x[p] * eps12[ceil_x[p], np.int64(y[p]), np.int64(z[p])] \
           + x_minus_ceil_x[p] * eps12[floor_x[p], np.int64(y[p]), np.int64(z[p])]
      tens[p,1,0] = tens[p,0,1]
      tens[p,0,2] = x_minus_floor_x[p] * eps13[ceil_x[p], np.int64(y[p]), np.int64(z[p])] \
           + x_minus_ceil_x[p] * eps13[floor_x[p], np.int64(y[p]), np.int64(z[p])]
      tens[p,2,0] = tens[p,0,2]
      tens[p,1,1] = x_minus_floor_x[p] * eps22[ceil_x[p], np.int64(y[p]), np.int64(z[p])] \
           + x_minus_ceil_x[p] * eps22[floor_x[p], np.int64(y[p]), np.int64(z[p])]
      tens[p,1,2] = x_minus_floor_x[p] * eps23[ceil_x[p], np.int64(y[p]), np.int64(z[p])] \
           + x_minus_ceil_x[p] * eps23[floor_x[p], np.int64(y[p]), np.int64(z[p])]
      tens[p,2,1] = tens[p,1,2]
      tens[p,2,2] = x_minus_floor_x[p] * eps33[ceil_x[p], np.int64(y[p]), np.int64(z[p])] \
           + x_minus_ceil_x[p] * eps33[floor_x[p], np.int64(y[p]), np.int64(z[p])]
    elif x[p] != floor_x[p] and y[p] != floor_y[p] and z[p] == floor_z[p]:
      floor_x_floor_y = x_minus_floor_x[p] * y_minus_floor_y[p]
      floor_x_ceil_y = x_minus_floor_x[p] * y_minus_ceil_y[p]
      ceil_x_floor_y = x_minus_ceil_x[p] * y_minus_floor_y[p]
      ceil_x_ceil_y = x_minus_ceil_x[p] * y_minus_ceil_y[p]
      
      tens[p,0,0] = floor_x_floor_y * eps11[ceil_x[p], ceil_y[p], np.int64(z[p])] \
           + ceil_x_floor_y * eps11[floor_x[p], ceil_y[p], np.int64(z[p])] \
           + floor_x_ceil_y * eps11[ceil_x[p], floor_y[p], np.int64(z[p])] \
           + ceil_x_ceil_y * eps11[floor_x[p], floor_y[p], np.int64(z[p])] 
      tens[p,0,1] = floor_x_floor_y * eps12[ceil_x[p], ceil_y[p], np.int64(z[p])] \
           + ceil_x_floor_y * eps12[floor_x[p], ceil_y[p], np.int64(z[p])] \
           + floor_x_ceil_y * eps12[ceil_x[p], floor_y[p], np.int64(z[p])] \
           + ceil_x_ceil_y * eps12[floor_x[p], floor_y[p], np.int64(z[p])] 
      tens[p,1,0] = tens[p,0,1]
      tens[p,0,2] = floor_x_floor_y * eps13[ceil_x[p], ceil_y[p], np.int64(z[p])] \
           + ceil_x_floor_y * eps13[floor_x[p], ceil_y[p], np.int64(z[p])] \
           + floor_x_ceil_y * eps13[ceil_x[p], floor_y[p], np.int64(z[p])] \
           + ceil_x_ceil_y * eps13[floor_x[p], floor_y[p], np.int64(z[p])] 
      tens[p,2,0] = tens[p,0,2]
      tens[p,1,1] = floor_x_floor_y * eps22[ceil_x[p], ceil_y[p], np.int64(z[p])] \
           + ceil_x_floor_y * eps22[floor_x[p], ceil_y[p], np.int64(z[p])] \
           + floor_x_ceil_y * eps22[ceil_x[p], floor_y[p], np.int64(z[p])] \
           + ceil_x_ceil_y * eps22[floor_x[p], floor_y[p], np.int64(z[p])] 
      tens[p,1,2] = floor_x_floor_y * eps23[ceil_x[p], ceil_y[p], np.int64(z[p])] \
           + ceil_x_floor_y * eps23[floor_x[p], ceil_y[p], np.int64(z[p])] \
           + floor_x_ceil_y * eps23[ceil_x[p], floor_y[p], np.int64(z[p])] \
           + ceil_x_ceil_y * eps23[floor_x[p], floor_y[p], np.int64(z[p])] 
      tens[p,2,1] = tens[p,1,2]
      tens[p,2,2] = floor_x_floor_y * eps33[ceil_x[p], ceil_y[p], np.int64(z[p])] \
           + ceil_x_floor_y * eps33[floor_x[p], ceil_y[p], np.int64(z[p])] \
           + floor_x_ceil_y * eps33[ceil_x[p], floor_y[p], np.int64(z[p])] \
           + ceil_x_ceil_y * eps33[floor_x[p], floor_y[p], np.int64(z[p])] 
    elif x[p] == floor_x[p] and y[p] != floor_y[p] and z[p] == floor_z[p]:
      tens[p,0,0] = y_minus_floor_y[p] * eps11[np.int64(x[p]), ceil_y[p], np.int64(z[p])] \
           + y_minus_ceil_y[p] * eps11[np.int64(x[p]), floor_y[p], np.int64(z[p])]
      tens[p,0,1] = y_minus_floor_y[p] * eps12[np.int64(x[p]), ceil_y[p], np.int64(z[p])] \
           + y_minus_ceil_y[p] * eps12[np.int64(x[p]), floor_y[p], np.int64(z[p])]
      tens[p,1,0] = tens[p,0,1]
      tens[p,0,2] = y_minus_floor_y[p] * eps13[np.int64(x[p]), ceil_y[p], np.int64(z[p])] \
           + y_minus_ceil_y[p] * eps13[np.int64(x[p]), floor_y[p], np.int64(z[p])]
      tens[p,2,0] = tens[p,0,2]
      tens[p,1,1] = y_minus_floor_y[p] * eps22[np.int64(x[p]), ceil_y[p], np.int64(z[p])] \
           + y_minus_ceil_y[p] * eps22[np.int64(x[p]), floor_y[p], np.int64(z[p])]
      tens[p,1,2] = y_minus_floor_y[p] * eps23[np.int64(x[p]), ceil_y[p], np.int64(z[p])] \
           + y_minus_ceil_y[p] * eps23[np.int64(x[p]), floor_y[p], np.int64(z[p])]
      tens[p,2,1] = tens[p,1,2]
      tens[p,2,2] = y_minus_floor_y[p] * eps33[np.int64(x[p]), ceil_y[p], np.int64(z[p])] \
           + y_minus_ceil_y[p] * eps33[np.int64(x[p]), floor_y[p], np.int64(z[p])]
    else:
      floor_x_floor_y = x_minus_floor_x[p] * y_minus_floor_y[p]
      floor_x_floor_y_floor_z = floor_x_floor_y * z_minus_floor_z[p]
      floor_x_floor_y_ceil_z = floor_x_floor_y * z_minus_ceil_z[p]
      floor_x_ceil_y = x_minus_floor_x[p] * y_minus_ceil_y[p]
      floor_x_ceil_y_floor_z = floor_x_ceil_y * z_minus_floor_z[p]
      floor_x_ceil_y_ceil_z = floor_x_ceil_y * z_minus_ceil_z[p]
      ceil_x_floor_y = x_minus_ceil_x[p] * y_minus_floor_y[p]
      ceil_x_floor_y_floor_z = ceil_x_floor_y * z_minus_floor_z[p]
      ceil_x_floor_y_ceil_z = ceil_x_floor_y * z_minus_ceil_z[p]
      ceil_x_ceil_y = x_minus_ceil_x[p] * y_minus_ceil_y[p]
      ceil_x_ceil_y_floor_z = ceil_x_ceil_y * z_minus_floor_z[p]
      ceil_x_ceil_y_ceil_z = ceil_x_ceil_y * z_minus_ceil_z[p]

      tens[p,0,0] = floor_x_floor_y_floor_z * eps11[ceil_x[p], ceil_y[p], ceil_z[p]] \
           + floor_x_ceil_y_floor_z * eps11[ceil_x[p], floor_y[p], ceil_z[p]] \
           + ceil_x_floor_y_floor_z * eps11[floor_x[p], ceil_y[p], ceil_z[p]] \
           + ceil_x_ceil_y_floor_z * eps11[floor_x[p], floor_y[p], ceil_z[p]] \
           + floor_x_floor_y_ceil_z * eps11[ceil_x[p], ceil_y[p], floor_z[p]] \
           + floor_x_ceil_y_ceil_z * eps11[ceil_x[p], floor_y[p], floor_z[p]] \
           + ceil_x_floor_y_ceil_z * eps11[floor_x[p], ceil_y[p], floor_z[p]] \
           + ceil_x_ceil_y_ceil_z * eps11[floor_x[p], floor_y[p], floor_z[p]]
      tens[p,0,1] = floor_x_floor_y_floor_z * eps12[ceil_x[p], ceil_y[p], ceil_z[p]] \
           + floor_x_ceil_y_floor_z * eps12[ceil_x[p], floor_y[p], ceil_z[p]] \
           + ceil_x_floor_y_floor_z * eps12[floor_x[p], ceil_y[p], ceil_z[p]] \
           + ceil_x_ceil_y_floor_z * eps12[floor_x[p], floor_y[p], ceil_z[p]] \
           + floor_x_floor_y_ceil_z * eps12[ceil_x[p], ceil_y[p], floor_z[p]] \
           + floor_x_ceil_y_ceil_z * eps12[ceil_x[p], floor_y[p], floor_z[p]] \
           + ceil_x_floor_y_ceil_z * eps12[floor_x[p], ceil_y[p], floor_z[p]] \
           + ceil_x_ceil_y_ceil_z * eps12[floor_x[p], floor_y[p], floor_z[p]]
      tens[p,1,0] = tens[p,0,1]
      tens[p,0,2] = floor_x_floor_y_floor_z * eps13[ceil_x[p], ceil_y[p], ceil_z[p]] \
           + floor_x_ceil_y_floor_z * eps13[ceil_x[p], floor_y[p], ceil_z[p]] \
           + ceil_x_floor_y_floor_z * eps13[floor_x[p], ceil_y[p], ceil_z[p]] \
           + ceil_x_ceil_y_floor_z * eps13[floor_x[p], floor_y[p], ceil_z[p]] \
           + floor_x_floor_y_ceil_z * eps13[ceil_x[p], ceil_y[p], floor_z[p]] \
           + floor_x_ceil_y_ceil_z * eps13[ceil_x[p], floor_y[p], floor_z[p]] \
           + ceil_x_floor_y_ceil_z * eps13[floor_x[p], ceil_y[p], floor_z[p]] \
           + ceil_x_ceil_y_ceil_z * eps13[floor_x[p], floor_y[p], floor_z[p]]
      tens[p,2,0] = tens[p,0,2]
      tens[p,1,1] = floor_x_floor_y_floor_z * eps22[ceil_x[p], ceil_y[p], ceil_z[p]] \
           + floor_x_ceil_y_floor_z * eps22[ceil_x[p], floor_y[p], ceil_z[p]] \
           + ceil_x_floor_y_floor_z * eps22[floor_x[p], ceil_y[p], ceil_z[p]] \
           + ceil_x_ceil_y_floor_z * eps22[floor_x[p], floor_y[p], ceil_z[p]] \
           + floor_x_floor_y_ceil_z * eps22[ceil_x[p], ceil_y[p], floor_z[p]] \
           + floor_x_ceil_y_ceil_z * eps22[ceil_x[p], floor_y[p], floor_z[p]] \
           + ceil_x_floor_y_ceil_z * eps22[floor_x[p], ceil_y[p], floor_z[p]] \
           + ceil_x_ceil_y_ceil_z * eps22[floor_x[p], floor_y[p], floor_z[p]]
      tens[p,1,2] = floor_x_floor_y_floor_z * eps23[ceil_x[p], ceil_y[p], ceil_z[p]] \
           + floor_x_ceil_y_floor_z * eps23[ceil_x[p], floor_y[p], ceil_z[p]] \
           + ceil_x_floor_y_floor_z * eps23[floor_x[p], ceil_y[p], ceil_z[p]] \
           + ceil_x_ceil_y_floor_z * eps23[floor_x[p], floor_y[p], ceil_z[p]] \
           + floor_x_floor_y_ceil_z * eps23[ceil_x[p], ceil_y[p], floor_z[p]] \
           + floor_x_ceil_y_ceil_z * eps23[ceil_x[p], floor_y[p], floor_z[p]] \
           + ceil_x_floor_y_ceil_z * eps23[floor_x[p], ceil_y[p], floor_z[p]] \
           + ceil_x_ceil_y_ceil_z * eps23[floor_x[p], floor_y[p], floor_z[p]]
      tens[p,2,1] = tens[p,1,2]
      tens[p,2,2] = floor_x_floor_y_floor_z * eps33[ceil_x[p], ceil_y[p], ceil_z[p]] \
           + floor_x_ceil_y_floor_z * eps33[ceil_x[p], floor_y[p], ceil_z[p]] \
           + ceil_x_floor_y_floor_z * eps33[floor_x[p], ceil_y[p], ceil_z[p]] \
           + ceil_x_ceil_y_floor_z * eps33[floor_x[p], floor_y[p], ceil_z[p]] \
           + floor_x_floor_y_ceil_z * eps33[ceil_x[p], ceil_y[p], floor_z[p]] \
           + floor_x_ceil_y_ceil_z * eps33[ceil_x[p], floor_y[p], floor_z[p]] \
           + ceil_x_floor_y_ceil_z * eps33[floor_x[p], ceil_y[p], floor_z[p]] \
           + ceil_x_ceil_y_ceil_z * eps33[floor_x[p], floor_y[p], floor_z[p]]
  
  return (tens)
# end batch_tens_interp_3d

def batch_tens_interp_3d_torch(x, y, z, tensor_field):
  num_tens = x.shape[0]
  tens = torch.zeros((num_tens, 3, 3),dtype=tensor_field.dtype)
  eps11 = tensor_field[0, :, :, :]
  eps12 = tensor_field[1, :, :, :]
  eps13 = tensor_field[2, :, :, :]
  eps22 = tensor_field[3, :, :, :]
  eps23 = tensor_field[4, :, :, :]
  eps33 = tensor_field[5, :, :, :]

  try:
    # Want to do torch.where(x<0,0,x), but get strange type promotion errors ala
    # https://github.com/pytorch/pytorch/issues/9190
    # hence the messy torch.tensor syntax
    x = torch.where(x<0,torch.tensor(0,dtype=x.dtype),x)
    x = torch.where(x>=eps11.shape[0]-1,torch.tensor(eps11.shape[0]-1,dtype=x.dtype),x)
    y = torch.where(y<0,torch.tensor(0,dtype=y.dtype),y)
    y = torch.where(y>=eps11.shape[1]-1,torch.tensor(eps11.shape[1]-1,dtype=y.dtype),y)
    z = torch.where(z<0,torch.tensor(0,dtype=z.dtype),z)
    z = torch.where(z>=eps11.shape[2]-1,torch.tensor(eps11.shape[2]-1,dtype=z.dtype),z)
  except Exception as err:
    print('Caught Exception:', err)
    print('x dtype',x.dtype)
    print('torch.where(x<0,0,x).dtype', torch.where(x<0,0,x).dtype)
    raise

  # Casting to double here because of this torch issue
  # https://github.com/pytorch/pytorch/issues/51199
  ceil_x = torch.ceil(x.double()).long()
  floor_x = torch.floor(x.double()).long()
  ceil_y = torch.ceil(y.double()).long()
  floor_y = torch.floor(y.double()).long()
  ceil_z = torch.ceil(z.double()).long()
  floor_z = torch.floor(z.double()).long()
  x_minus_floor_x = torch.abs(x - floor_x)
  x_minus_ceil_x = torch.abs(x - ceil_x)
  y_minus_floor_y = torch.abs(y - floor_y)
  y_minus_ceil_y = torch.abs(y - ceil_y)
  z_minus_floor_z = torch.abs(z - floor_z)
  z_minus_ceil_z = torch.abs(z - ceil_z)

  # Find index where interpolation is needed and interpolate
  intidx = torch.where((x_minus_floor_x + x_minus_ceil_x
               + y_minus_floor_y + y_minus_ceil_y
               + z_minus_floor_z + z_minus_ceil_z) >= 1e-14)
  try:
    if len(intidx[0]) > 0:
      tens[intidx[0][:],0,0] = x_minus_floor_x[intidx] * y_minus_floor_y[intidx] * z_minus_floor_z[intidx] * eps11[ceil_x[intidx], ceil_y[intidx], ceil_z[intidx]] \
           + x_minus_floor_x[intidx] * y_minus_ceil_y[intidx] * z_minus_floor_z[intidx] * eps11[ceil_x[intidx], floor_y[intidx], ceil_z[intidx]] \
           + x_minus_ceil_x[intidx] * y_minus_floor_y[intidx] * z_minus_floor_z[intidx] * eps11[floor_x[intidx], ceil_y[intidx], ceil_z[intidx]] \
           + x_minus_ceil_x[intidx] * y_minus_ceil_y[intidx] * z_minus_floor_z[intidx] * eps11[floor_x[intidx], floor_y[intidx], ceil_z[intidx]] \
           + x_minus_floor_x[intidx] * y_minus_floor_y[intidx] * z_minus_ceil_z[intidx] * eps11[ceil_x[intidx], ceil_y[intidx], floor_z[intidx]] \
           + x_minus_floor_x[intidx] * y_minus_ceil_y[intidx] * z_minus_ceil_z[intidx] * eps11[ceil_x[intidx], floor_y[intidx], floor_z[intidx]] \
           + x_minus_ceil_x[intidx] * y_minus_floor_y[intidx] * z_minus_ceil_z[intidx] * eps11[floor_x[intidx], ceil_y[intidx], floor_z[intidx]] \
           + x_minus_ceil_x[intidx] * y_minus_ceil_y[intidx] * z_minus_ceil_z[intidx] * eps11[floor_x[intidx], floor_y[intidx], floor_z[intidx]]
      tens[intidx[0][:],0,1] = x_minus_floor_x[intidx] * y_minus_floor_y[intidx] * z_minus_floor_z[intidx] * eps12[ceil_x[intidx], ceil_y[intidx], ceil_z[intidx]] \
           + x_minus_floor_x[intidx] * y_minus_ceil_y[intidx] * z_minus_floor_z[intidx] * eps12[ceil_x[intidx], floor_y[intidx], ceil_z[intidx]] \
           + x_minus_ceil_x[intidx] * y_minus_floor_y[intidx] * z_minus_floor_z[intidx] * eps12[floor_x[intidx], ceil_y[intidx], ceil_z[intidx]] \
           + x_minus_ceil_x[intidx] * y_minus_ceil_y[intidx] * z_minus_floor_z[intidx] * eps12[floor_x[intidx], floor_y[intidx], ceil_z[intidx]] \
           + x_minus_floor_x[intidx] * y_minus_floor_y[intidx] * z_minus_ceil_z[intidx] * eps12[ceil_x[intidx], ceil_y[intidx], floor_z[intidx]] \
           + x_minus_floor_x[intidx] * y_minus_ceil_y[intidx] * z_minus_ceil_z[intidx] * eps12[ceil_x[intidx], floor_y[intidx], floor_z[intidx]] \
           + x_minus_ceil_x[intidx] * y_minus_floor_y[intidx] * z_minus_ceil_z[intidx] * eps12[floor_x[intidx], ceil_y[intidx], floor_z[intidx]] \
           + x_minus_ceil_x[intidx] * y_minus_ceil_y[intidx] * z_minus_ceil_z[intidx] * eps12[floor_x[intidx], floor_y[intidx], floor_z[intidx]]
      tens[intidx[0][:],1,0] = tens[intidx[0][:],0,1]
      tens[intidx[0][:],0,2] = x_minus_floor_x[intidx] * y_minus_floor_y[intidx] * z_minus_floor_z[intidx] * eps13[ceil_x[intidx], ceil_y[intidx], ceil_z[intidx]] \
           + x_minus_floor_x[intidx] * y_minus_ceil_y[intidx] * z_minus_floor_z[intidx] * eps13[ceil_x[intidx], floor_y[intidx], ceil_z[intidx]] \
           + x_minus_ceil_x[intidx] * y_minus_floor_y[intidx] * z_minus_floor_z[intidx] * eps13[floor_x[intidx], ceil_y[intidx], ceil_z[intidx]] \
           + x_minus_ceil_x[intidx] * y_minus_ceil_y[intidx] * z_minus_floor_z[intidx] * eps13[floor_x[intidx], floor_y[intidx], ceil_z[intidx]] \
           + x_minus_floor_x[intidx] * y_minus_floor_y[intidx] * z_minus_ceil_z[intidx] * eps13[ceil_x[intidx], ceil_y[intidx], floor_z[intidx]] \
           + x_minus_floor_x[intidx] * y_minus_ceil_y[intidx] * z_minus_ceil_z[intidx] * eps13[ceil_x[intidx], floor_y[intidx], floor_z[intidx]] \
           + x_minus_ceil_x[intidx] * y_minus_floor_y[intidx] * z_minus_ceil_z[intidx] * eps13[floor_x[intidx], ceil_y[intidx], floor_z[intidx]] \
           + x_minus_ceil_x[intidx] * y_minus_ceil_y[intidx] * z_minus_ceil_z[intidx] * eps13[floor_x[intidx], floor_y[intidx], floor_z[intidx]]
      tens[intidx[0][:],2,0] = tens[intidx[0][:],0,2]
      tens[intidx[0][:],1,1] = x_minus_floor_x[intidx] * y_minus_floor_y[intidx] * z_minus_floor_z[intidx] * eps22[ceil_x[intidx], ceil_y[intidx], ceil_z[intidx]] \
           + x_minus_floor_x[intidx] * y_minus_ceil_y[intidx] * z_minus_floor_z[intidx] * eps22[ceil_x[intidx], floor_y[intidx], ceil_z[intidx]] \
           + x_minus_ceil_x[intidx] * y_minus_floor_y[intidx] * z_minus_floor_z[intidx] * eps22[floor_x[intidx], ceil_y[intidx], ceil_z[intidx]] \
           + x_minus_ceil_x[intidx] * y_minus_ceil_y[intidx] * z_minus_floor_z[intidx] * eps22[floor_x[intidx], floor_y[intidx], ceil_z[intidx]] \
           + x_minus_floor_x[intidx] * y_minus_floor_y[intidx] * z_minus_ceil_z[intidx] * eps22[ceil_x[intidx], ceil_y[intidx], floor_z[intidx]] \
           + x_minus_floor_x[intidx] * y_minus_ceil_y[intidx] * z_minus_ceil_z[intidx] * eps22[ceil_x[intidx], floor_y[intidx], floor_z[intidx]] \
           + x_minus_ceil_x[intidx] * y_minus_floor_y[intidx] * z_minus_ceil_z[intidx] * eps22[floor_x[intidx], ceil_y[intidx], floor_z[intidx]] \
           + x_minus_ceil_x[intidx] * y_minus_ceil_y[intidx] * z_minus_ceil_z[intidx] * eps22[floor_x[intidx], floor_y[intidx], floor_z[intidx]]
      tens[intidx[0][:],1,2] = x_minus_floor_x[intidx] * y_minus_floor_y[intidx] * z_minus_floor_z[intidx] * eps23[ceil_x[intidx], ceil_y[intidx], ceil_z[intidx]] \
           + x_minus_floor_x[intidx] * y_minus_ceil_y[intidx] * z_minus_floor_z[intidx] * eps23[ceil_x[intidx], floor_y[intidx], ceil_z[intidx]] \
           + x_minus_ceil_x[intidx] * y_minus_floor_y[intidx] * z_minus_floor_z[intidx] * eps23[floor_x[intidx], ceil_y[intidx], ceil_z[intidx]] \
           + x_minus_ceil_x[intidx] * y_minus_ceil_y[intidx] * z_minus_floor_z[intidx] * eps23[floor_x[intidx], floor_y[intidx], ceil_z[intidx]] \
           + x_minus_floor_x[intidx] * y_minus_floor_y[intidx] * z_minus_ceil_z[intidx] * eps23[ceil_x[intidx], ceil_y[intidx], floor_z[intidx]] \
           + x_minus_floor_x[intidx] * y_minus_ceil_y[intidx] * z_minus_ceil_z[intidx] * eps23[ceil_x[intidx], floor_y[intidx], floor_z[intidx]] \
           + x_minus_ceil_x[intidx] * y_minus_floor_y[intidx] * z_minus_ceil_z[intidx] * eps23[floor_x[intidx], ceil_y[intidx], floor_z[intidx]] \
           + x_minus_ceil_x[intidx] * y_minus_ceil_y[intidx] * z_minus_ceil_z[intidx] * eps23[floor_x[intidx], floor_y[intidx], floor_z[intidx]]
      tens[intidx[0][:],2,1] = tens[intidx[0][:],1,2]
      tens[intidx[0][:],2,2] = x_minus_floor_x[intidx] * y_minus_floor_y[intidx] * z_minus_floor_z[intidx] * eps33[ceil_x[intidx], ceil_y[intidx], ceil_z[intidx]] \
           + x_minus_floor_x[intidx] * y_minus_ceil_y[intidx] * z_minus_floor_z[intidx] * eps33[ceil_x[intidx], floor_y[intidx], ceil_z[intidx]] \
           + x_minus_ceil_x[intidx] * y_minus_floor_y[intidx] * z_minus_floor_z[intidx] * eps33[floor_x[intidx], ceil_y[intidx], ceil_z[intidx]] \
           + x_minus_ceil_x[intidx] * y_minus_ceil_y[intidx] * z_minus_floor_z[intidx] * eps33[floor_x[intidx], floor_y[intidx], ceil_z[intidx]] \
           + x_minus_floor_x[intidx] * y_minus_floor_y[intidx] * z_minus_ceil_z[intidx] * eps33[ceil_x[intidx], ceil_y[intidx], floor_z[intidx]] \
           + x_minus_floor_x[intidx] * y_minus_ceil_y[intidx] * z_minus_ceil_z[intidx] * eps33[ceil_x[intidx], floor_y[intidx], floor_z[intidx]] \
           + x_minus_ceil_x[intidx] * y_minus_floor_y[intidx] * z_minus_ceil_z[intidx] * eps33[floor_x[intidx], ceil_y[intidx], floor_z[intidx]] \
           + x_minus_ceil_x[intidx] * y_minus_ceil_y[intidx] * z_minus_ceil_z[intidx] * eps33[floor_x[intidx], floor_y[intidx], floor_z[intidx]]
  except Exception as err:
    print('Caught Exception:', err)
    print('intidx:',intidx)
    print(intidx[0].shape)
    print(torch.sum(x_minus_floor_x[intidx]))
    print(torch.sum(eps11[ceil_x[intidx], ceil_y[intidx], ceil_z[intidx]]))
    raise

  # Find index where no interpolation is needed and just copy the values
  nointidx = torch.where((x_minus_floor_x + x_minus_ceil_x
               + y_minus_floor_y + y_minus_ceil_y
               + z_minus_floor_z + z_minus_ceil_z) < 1e-14)

  try:
    if len(nointidx[0]) > 0:
      # Since x == floor_x, y == floor_y and z == floor_z in this case, use them to index to avoid type error
      # IndexError: tensors used as indices must be long, byte or bool tensors
      tens[nointidx[0][:],0,0] = eps11[floor_x[nointidx],floor_y[nointidx],floor_z[nointidx]]
      tens[nointidx[0][:],0,1] = eps12[floor_x[nointidx],floor_y[nointidx],floor_z[nointidx]]
      tens[nointidx[0][:],1,0] = tens[nointidx[0][:],0,1]
      tens[nointidx[0][:],0,2] = eps13[floor_x[nointidx],floor_y[nointidx],floor_z[nointidx]]
      tens[nointidx[0][:],2,0] = tens[nointidx[0][:],0,2]
      tens[nointidx[0][:],1,1] = eps22[floor_x[nointidx],floor_y[nointidx],floor_z[nointidx]]
      tens[nointidx[0][:],1,2] = eps23[floor_x[nointidx],floor_y[nointidx],floor_z[nointidx]]
      tens[nointidx[0][:],2,1] = tens[nointidx[0][:],1,2]    
      tens[nointidx[0][:],2,2] = eps33[floor_x[nointidx],floor_y[nointidx],floor_z[nointidx]]
  except Exception as err:
    print('Caught Exception:', err)
    print('nointidx:',nointidx)
    print(nointidx[0].shape)
    print(torch.sum(x_minus_floor_x[nointidx]))
    print(torch.sum(eps11[ceil_x[nointidx], ceil_y[nointidx], ceil_z[nointidx]]))
    raise

  return (tens)

def vect_interp_3d(x, y, z, vector_field):
  vect = np.zeros((3))
  v1 = vector_field[0]
  v2 = vector_field[1]
  v3 = vector_field[2]
  if ((math.floor(x) < 0) or (math.ceil(x) >= v1.shape[0])
      or (math.floor(y) < 0) or (math.ceil(y) >= v1.shape[1])
      or (math.floor(z) < 0) or (math.ceil(z) >= v1.shape[2])):
     # data is out of bounds, return identity vector
     vect[0] = 1
     vect[1] = 1
     vect[2] = 1
     return(vect)

   
  if x == math.floor(x) and y == math.floor(y) and z == math.floor(z):
    vect[0] = v1[int(x), int(y), int(z)]
    vect[1] = v2[int(x), int(y), int(z)]
    vect[2] = v3[int(x), int(y), int(z)]
  elif x == math.floor(x) and y != math.floor(y) and z != math.floor(z):
    vect[0] = abs(y - math.floor(y)) * abs(z - math.floor(z)) * v1[int(x), math.ceil(y), math.ceil(z)] \
                 + abs(y - math.ceil(y)) * abs(z - math.floor(z)) * v1[int(x), math.floor(y), math.ceil(z)] \
                 + abs(y - math.floor(y)) * abs(z - math.ceil(z)) * v1[int(x), math.ceil(y), math.floor(z)] \
                 + abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * v1[int(x), math.floor(y), math.floor(z)] 
    vect[1] = abs(y - math.floor(y)) * abs(z - math.floor(z)) * v2[int(x), math.ceil(y), math.ceil(z)] \
                 + abs(y - math.ceil(y)) * abs(z - math.floor(z)) * v2[int(x), math.floor(y), math.ceil(z)] \
                 + abs(y - math.floor(y)) * abs(z - math.ceil(z)) * v2[int(x), math.ceil(y), math.floor(z)] \
                 + abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * v2[int(x), math.floor(y), math.floor(z)] 
    vect[2] = abs(y - math.floor(y)) * abs(z - math.floor(z)) * v3[int(x), math.ceil(y), math.ceil(z)] \
                 + abs(y - math.ceil(y)) * abs(z - math.floor(z)) * v3[int(x), math.floor(y), math.ceil(z)] \
                 + abs(y - math.floor(y)) * abs(z - math.ceil(z)) * v3[int(x), math.ceil(y), math.floor(z)] \
                 + abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * v3[int(x), math.floor(y), math.floor(z)] 
    
  elif x == math.floor(x) and y == math.floor(y) and z != math.floor(z):
    vect[0] = abs(z - math.floor(z)) * v1[int(x), int(y), math.ceil(z)] \
                 + abs(z - math.ceil(z)) * v1[int(x), int(y), math.floor(z)] 
    vect[1] = abs(z - math.floor(z)) * v2[int(x), int(y), math.ceil(z)] \
                 + abs(z - math.ceil(z)) * v2[int(x), int(y), math.floor(z)] 
    vect[2] = abs(z - math.floor(z)) * v3[int(x), int(y), math.ceil(z)] \
                 + abs(z - math.ceil(z)) * v3[int(x), int(y), math.floor(z)] 
  elif x != math.floor(x) and y == math.floor(y) and z != math.floor(z):
    vect[0] = abs(x - math.floor(x)) * abs(z - math.floor(z)) * v1[math.ceil(x), int(y), math.ceil(z)] \
                 + abs(x - math.ceil(x)) * abs(z - math.floor(z)) * v1[math.floor(x), int(y), math.ceil(z)] \
                 + abs(x - math.floor(x)) * abs(z - math.ceil(z)) * v1[math.ceil(x), int(y), math.floor(z)] \
                 + abs(x - math.ceil(x)) * abs(z - math.ceil(z)) * v1[math.floor(x), int(y), math.floor(z)]
    vect[1] = abs(x - math.floor(x)) * abs(z - math.floor(z)) * v2[math.ceil(x), int(y), math.ceil(z)] \
                 + abs(x - math.ceil(x)) * abs(z - math.floor(z)) * v2[math.floor(x), int(y), math.ceil(z)] \
                 + abs(x - math.floor(x)) * abs(z - math.ceil(z)) * v2[math.ceil(x), int(y), math.floor(z)] \
                 + abs(x - math.ceil(x)) * abs(z - math.ceil(z)) * v2[math.floor(x), int(y), math.floor(z)]
    vect[2] = abs(x - math.floor(x)) * abs(z - math.floor(z)) * v3[math.ceil(x), int(y), math.ceil(z)] \
                 + abs(x - math.ceil(x)) * abs(z - math.floor(z)) * v3[math.floor(x), int(y), math.ceil(z)] \
                 + abs(x - math.floor(x)) * abs(z - math.ceil(z)) * v3[math.ceil(x), int(y), math.floor(z)] \
                 + abs(x - math.ceil(x)) * abs(z - math.ceil(z)) * v3[math.floor(x), int(y), math.floor(z)]
  elif x != math.floor(x) and y == math.floor(y) and z == math.floor(z):
    vect[0] = abs(x - math.floor(x)) * v1[math.ceil(x), int(y),int(z)] \
                 + abs(x - math.ceil(x)) * v1[math.floor(x), int(y), int(z)]
    vect[1] = abs(x - math.floor(x)) * v2[math.ceil(x), int(y),int(z)] \
                 + abs(x - math.ceil(x)) * v2[math.floor(x), int(y), int(z)]
    vect[2] = abs(x - math.floor(x)) * v3[math.ceil(x), int(y),int(z)] \
                 + abs(x - math.ceil(x)) * v3[math.floor(x), int(y), int(z)]
  elif x != math.floor(x) and y != math.floor(y) and z == math.floor(z):
    vect[0] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * v1[math.ceil(x), math.ceil(y), int(z)] \
                 + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * v1[math.floor(x), math.ceil(y), int(z)] \
                 + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * v1[math.ceil(x), math.floor(y), int(z)] \
                 + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * v1[math.floor(x), math.floor(y), int(z)]  
    vect[1] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * v2[math.ceil(x), math.ceil(y), int(z)] \
                 + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * v2[math.floor(x), math.ceil(y), int(z)] \
                 + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * v2[math.ceil(x), math.floor(y), int(z)] \
                 + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * v2[math.floor(x), math.floor(y), int(z)]  
    vect[2] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * v3[math.ceil(x), math.ceil(y), int(z)] \
                 + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * v3[math.floor(x), math.ceil(y), int(z)] \
                 + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * v3[math.ceil(x), math.floor(y), int(z)] \
                 + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * v3[math.floor(x), math.floor(y), int(z)]  
  elif x == math.floor(x) and y != math.floor(y) and z == math.floor(z):
    vect[0] = abs(y - math.floor(y)) * v1[int(x), math.ceil(y), int(z)] \
                 + abs(y - math.ceil(y)) * v1[int(x), math.floor(y), int(z)]
    vect[1] = abs(y - math.floor(y)) * v2[int(x), math.ceil(y), int(z)] \
                 + abs(y - math.ceil(y)) * v2[int(x), math.floor(y), int(z)]
    vect[2] = abs(y - math.floor(y)) * v3[int(x), math.ceil(y), int(z)] \
                 + abs(y - math.ceil(y)) * v3[int(x), math.floor(y), int(z)]
  else:
    vect[0] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * abs(z - math.floor(z)) * v1[math.ceil(x), math.ceil(y), math.ceil(z)] \
           + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * abs(z - math.floor(z)) * v1[math.ceil(x), math.floor(y), math.ceil(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * abs(z - math.floor(z)) * v1[math.floor(x), math.ceil(y), math.ceil(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * abs(z - math.floor(z)) * v1[math.floor(x), math.floor(y), math.ceil(z)] \
           + abs(x - math.floor(x)) * abs(y - math.floor(y)) * abs(z - math.ceil(z)) * v1[math.ceil(x), math.ceil(y), math.floor(z)] \
           + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * v1[math.ceil(x), math.floor(y), math.floor(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * abs(z - math.ceil(z)) * v1[math.floor(x), math.ceil(y), math.floor(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * v1[math.floor(x), math.floor(y), math.floor(z)]
    vect[1] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * abs(z - math.floor(z)) * v2[math.ceil(x), math.ceil(y), math.ceil(z)] \
           + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * abs(z - math.floor(z)) * v2[math.ceil(x), math.floor(y), math.ceil(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * abs(z - math.floor(z)) * v2[math.floor(x), math.ceil(y), math.ceil(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * abs(z - math.floor(z)) * v2[math.floor(x), math.floor(y), math.ceil(z)] \
           + abs(x - math.floor(x)) * abs(y - math.floor(y)) * abs(z - math.ceil(z)) * v2[math.ceil(x), math.ceil(y), math.floor(z)] \
           + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * v2[math.ceil(x), math.floor(y), math.floor(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * abs(z - math.ceil(z)) * v2[math.floor(x), math.ceil(y), math.floor(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * v2[math.floor(x), math.floor(y), math.floor(z)]
    vect[2] = abs(x - math.floor(x)) * abs(y - math.floor(y)) * abs(z - math.floor(z)) * v3[math.ceil(x), math.ceil(y), math.ceil(z)] \
           + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * abs(z - math.floor(z)) * v3[math.ceil(x), math.floor(y), math.ceil(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * abs(z - math.floor(z)) * v3[math.floor(x), math.ceil(y), math.ceil(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * abs(z - math.floor(z)) * v3[math.floor(x), math.floor(y), math.ceil(z)] \
           + abs(x - math.floor(x)) * abs(y - math.floor(y)) * abs(z - math.ceil(z)) * v3[math.ceil(x), math.ceil(y), math.floor(z)] \
           + abs(x - math.floor(x)) * abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * v3[math.ceil(x), math.floor(y), math.floor(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.floor(y)) * abs(z - math.ceil(z)) * v3[math.floor(x), math.ceil(y), math.floor(z)] \
           + abs(x - math.ceil(x)) * abs(y - math.ceil(y)) * abs(z - math.ceil(z)) * v3[math.floor(x), math.floor(y), math.floor(z)]
  return (vect)