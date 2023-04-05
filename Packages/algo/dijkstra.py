from lazy_imports import np
from data import io

class Adjacency:
    def __init__(self, a, b, dist):
        self.a = a
        self.b = b
        self.dist = dist


class Edge:
    def __init__(self, node, next, weight):
        self.node = node
        self.next = next
        self.weight = weight

class Vertex:
    def __init__(self, idx, x, y, neighbors, dists):
        # dists is the dist to each associated neighbor
        self.idx = idx
        self.x = x
        self.y = y
        self.neighbors = neighbors
        self.dists = dists
        self.parent = None
        self.visited = False
        self.dist_to_start = np.inf

def const_vertex_list(mask, eps11, eps12, eps22):
  #     inverse
  eps_11 = eps22 / (eps11*eps22 - eps12*eps12)
  eps_12 = -eps12 / (eps11*eps22 - eps12*eps12)
  eps_22 = eps11 / (eps11*eps22 - eps12*eps12)

  #eps_11 = eps11
  #eps_12 = eps12
  #eps_22 = eps22
  
  verts = {}
  #neighbor = np.array([[-1, -1],
  #                    [-1, 0],
  #                    [-1, 1],
  #                    [0, -1],
  #                    [0, 1],
  #                    [1, -1],
  #                    [1, 0],
  #                    [1, 1]])
  neighbor = np.array([[-1, 0],
                       [0, -1],
                       [0, 1],
                       [1, 0],
                       [-1, -1],
                       [-1, 1],
                       [1, -1],
                       [1, 1]])

  for x in range(mask.shape[0]):
    for y in range(mask.shape[1]):
      if mask[x,y]:
        idx = np.ravel_multi_index([x,y],mask.shape)
        ns = []
        dists = []
        for j in range(neighbor.shape[0]):
          if mask[x+neighbor[j, 0], y+neighbor[j, 1]]:
            eps11_itp = (eps_11[x + neighbor[j, 0], y + neighbor[j, 1]] +
                         eps_11[x, y]) / 2
            eps12_itp = (eps_12[x + neighbor[j, 0], y + neighbor[j, 1]] +
                         eps_12[x, y]) / 2
            eps22_itp = (eps_22[x + neighbor[j, 0], y + neighbor[j, 1]] +
                         eps_22[x, y]) / 2
            ns.append(np.ravel_multi_index([x + neighbor[j,0], y + neighbor[j,1]], mask.shape))
            dists.append(np.sqrt(neighbor[j, 0] ** 2 * eps11_itp + \
                                 2 * neighbor[j, 0] * neighbor[j, 1] * eps12_itp + \
                                 neighbor[j, 1] ** 2 * eps22_itp))
        verts[idx] = Vertex(idx, x, y, ns, dists)
  return(verts)
    
def const_adj_list(vertex, target_area, eps11, eps12, eps22, x, y, v):
    e = 0
    neighbor = np.array([[-1, -1],
                        [-1, 0],
                        [-1, 1],
                        [0, -1],
                        [0, 1],
                        [1, -1],
                        [1, 0],
                        [1, 1]])
    #neighbor = np.array([[-1, 0],
    #                    [0, -1],
    #                    [0, 1],
    #                    [1, 0]])
    adjacency_list = []

    #     inverse
    eps_11 = eps22 / (eps11*eps22 - eps12*eps12)
    eps_12 = -eps12 / (eps11*eps22 - eps12*eps12)
    eps_22 = eps11 / (eps11*eps22 - eps12*eps12)

    for i in range(v):
        for j in range(neighbor.shape[0]):
            #             inverse
            if target_area[vertex[i, 0]+neighbor[j, 0], vertex[i, 1]+neighbor[j, 1]]==1:
                eps11_itp = (eps_11[vertex[i, 0] + neighbor[j, 0], vertex[i, 1] + neighbor[j, 1]] +
                             eps_11[vertex[i, 0], vertex[i, 1]]) / 2
                eps12_itp = (eps_12[vertex[i, 0] + neighbor[j, 0], vertex[i, 1] + neighbor[j, 1]] +
                             eps_12[vertex[i, 0], vertex[i, 1]]) / 2
                eps22_itp = (eps_22[vertex[i, 0] + neighbor[j, 0], vertex[i, 1] + neighbor[j, 1]] +
                             eps_22[vertex[i, 0], vertex[i, 1]]) / 2
                #eps11_itp = eps_11[vertex[i, 0], vertex[i, 1]]
                #eps12_itp = eps_12[vertex[i, 0], vertex[i, 1]]
                #eps22_itp = eps_22[vertex[i, 0], vertex[i, 1]]

                #dist = neighbor[j, 0] ** 2 * eps11_itp + \
                #       2 * neighbor[j, 0] * neighbor[j, 1] * eps12_itp + \
                #       neighbor[j, 1] ** 2 * eps22_itp
                dist = np.sqrt(neighbor[j, 0] ** 2 * eps11_itp + \
                       2 * neighbor[j, 0] * neighbor[j, 1] * eps12_itp + \
                       neighbor[j, 1] ** 2 * eps22_itp)

                nx = np.argwhere(x == (vertex[i, 0] + neighbor[j, 0]))
                ny = np.argwhere(y == (vertex[i, 1] + neighbor[j, 1]))
                n = np.intersect1d(nx, ny)

                adjacency_list.append(Adjacency(int(i), int(n[0]), dist))

                e = e + 1

    return adjacency_list, e


def const_edge_list(adjacency_list, e, v):
    first_arc = -1 * np.ones(v, dtype=int)
    edge_list = []
    #     construct edge_list
    for i in range(e):
        edge_instance = Edge(adjacency_list[i].b, first_arc[adjacency_list[i].a], adjacency_list[i].dist)
        edge_list.append(edge_instance)
        first_arc[adjacency_list[i].a] = i

    return edge_list, first_arc


def const_path(start, v, visited, predecessor):
    path = np.zeros((v, v), dtype=int)
    for i in range(v):
        if visited[i] == 1 and i != start:
            out_cnt = 1
            path[i, out_cnt] = i
            t = predecessor[i]
            if t == -1:
                print('no path')
            else:
                while t != start:
                    out_cnt = out_cnt + 1
                    path[i, out_cnt] = t
                    t = predecessor[t]

                out_cnt = out_cnt + 1
                path[i, out_cnt] = start

    return path

def dijkstra(edge_list, first_arc, dist_from_origin, v, visited, predecessor):
    k = 1
    for i in range(v):
        min_dist = np.inf
        for j in range(v):
            if visited[j] == 0 and dist_from_origin[j] < min_dist:
                k = j
                min_dist = dist_from_origin[j]

        visited[k] = 1
        p = first_arc[k]
        while p != -1:
            t = edge_list[p].node
            if visited[t] == 0 and dist_from_origin[k] + edge_list[p].weight < dist_from_origin[t]:
                dist_from_origin[t] = dist_from_origin[k] + edge_list[p].weight
                predecessor[t] = k
            p = edge_list[p].next

    return predecessor, visited, dist_from_origin


def shortpath(tensor_field, mask, start_coordinate, end_coordinate, metric='', filename = ''):

    print(f"Finding shortest path from {start_coordinate} to {end_coordinate}")
    
    eps11 = tensor_field[0, :, :]
    eps12 = tensor_field[1, :, :]
    eps22 = tensor_field[2, :, :]

    # inverse
    #eps11 = np.where(eps11 == 1, 5e-1, eps11)
    #eps22 = np.where(eps22 == 1, 5e-1, eps22)

    # apply scaling field
    if metric=='withscaling':
        scaling_field = np.loadtxt(open("input/e_alpha_kris.csv", "rb"), delimiter=",")
        eps11 = eps11 * scaling_field
        eps12 = eps12 * scaling_field
        eps22 = eps22 * scaling_field

    v = int(mask.sum())

    # index every entries inside mask
    x, y = np.where(mask == 1)
    vertex = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1)), 1)

    # get index of the start and the end point
    idx_matchx = np.where(vertex[:,0]==start_coordinate[0])
    idx_matchy = np.where(vertex[:,1]==start_coordinate[1])
    idx_start = np.intersect1d(idx_matchx, idx_matchy)
    idx_start = idx_start[0]

    idx_matchx = np.where(vertex[:, 0] == end_coordinate[0])
    idx_matchy = np.where(vertex[:, 1] == end_coordinate[1])
    idx_end = np.intersect1d(idx_matchx, idx_matchy)
    idx_end = idx_end[0]

    # construct adjacency list
    adjacency_list, e = const_adj_list(vertex, mask, eps11, eps12, eps22, x, y, v)

    # construct edge list
    edge_list, first_arc = const_edge_list(adjacency_list, e, v)

    # calculate the shortest path
    dist_from_origin = np.full(v, np.inf)
    visited = np.zeros(v, dtype=int)
    visited[idx_start] = 1
    # predecessor = np.copy(first_arc)
    predecessor = np.zeros(v, dtype=int)
    p = first_arc[idx_start]
    while p != -1:
        dist_from_origin[edge_list[p].node] = edge_list[p].weight
        predecessor[edge_list[p].node] = idx_start
        p = edge_list[p].next

    predecessor[idx_start] = 0
    predecessor, visited, dist_from_origin = dijkstra(edge_list, first_arc, dist_from_origin, v, visited, predecessor)

    # construct path list
    path = const_path(idx_start, v, visited, predecessor)

    # display
    i = 0
    shortpath_points_x = []
    shortpath_points_y = []
    while path[idx_end, i] != idx_start:
        shortpath_points_x.append(int(vertex[path[idx_end, i], 0]))
        shortpath_points_y.append(int(vertex[path[idx_end, i], 1]))
        i = i + 1

    if filename:
      io.writePath(shortpath_points_x, shortpath_points_y, filename)
      
    return shortpath_points_x, shortpath_points_y, dist_from_origin

def dijkstra2(vertex_list):
  # assumes source vertex already has minimum distance set
  indices = list(vertex_list.keys())
  while len(indices) > 0:
    min_idx = -1
    min_i = -1
    min_dist = np.inf
    for i in range(len(indices)):
      idx = indices[i]
      if (not vertex_list[idx].visited) and vertex_list[idx].dist_to_start < min_dist:
        min_idx = idx
        min_i = i
        min_dist = vertex_list[idx].dist_to_start

    vert = vertex_list[min_idx]
    indices.pop(min_i)
    vert.visited = True
    for n,d in zip(vert.neighbors, vert.dists):
      if vert.dist_to_start + d < vertex_list[n].dist_to_start:
        vertex_list[n].dist_to_start = vert.dist_to_start + d
        vertex_list[n].parent = min_idx

  return()

  
def shortpath2(tensor_field, mask, start_coordinate, end_coordinate, filename = ''):

  print(f"Finding shortest path from {start_coordinate} to {end_coordinate}")
    
  eps11 = tensor_field[0, :, :]
  eps12 = tensor_field[1, :, :]
  eps22 = tensor_field[2, :, :]

  # inverse
  #eps11 = np.where(eps11 == 1, 5e-1, eps11)
  #eps22 = np.where(eps22 == 1, 5e-1, eps22)

  # construct vertices
  vertex_list = const_vertex_list(mask, eps11, eps12, eps22)

  # calculate the shortest path
  idx_start = np.ravel_multi_index(start_coordinate, mask.shape)
  idx_end = np.ravel_multi_index(end_coordinate, mask.shape)
    
  vertex_list[idx_start].dist_to_start = 0
    
  dijkstra2(vertex_list)

  # construct path list
  # skip constructing all possible paths, instead just do from idx_end
  #paths = const_path2(vertex_list)

  # display
  i = 0
  shortpath_points_x = []
  shortpath_points_y = []
  dist_to_start = vertex_list[idx_end].dist_to_start

  cur_idx = idx_end

  if not mask[end_coordinate[0], end_coordinate[1]]:
    print(f"end coordinate {end_coordinate} not in masked region")
    return shortpath_points_x, shortpath_points_y, dist_to_start
  
  while cur_idx != idx_start:
    try:
      shortpath_points_x.append(vertex_list[cur_idx].x)
      shortpath_points_y.append(vertex_list[cur_idx].y)
      cur_idx = vertex_list[cur_idx].parent
    except:
      print(f"error. {cur_idx}, {idx_start}")

  shortpath_points_x.append(vertex_list[idx_start].x)
  shortpath_points_y.append(vertex_list[idx_start].y)

  if filename:
      io.writePath(shortpath_points_x, shortpath_points_y, filename)

  return shortpath_points_x, shortpath_points_y, dist_to_start
