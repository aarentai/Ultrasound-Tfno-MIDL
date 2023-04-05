# tools for working with NRRD files
import os

from . import fileManager as fm

def find_assoc_nrrd_data_file(nhdr):
  # first see if this is a separated nhdr / data format
  # or a nrrd file that is header + data
  froot, fext = os.path.splitext(nhdr)
  if fext == ".nrrd":
    return ""
  # determine the name of the associated data file
  filename = ""
  try:
    fd = open(nhdr,'r')
    for line in fd:
      line = line.strip()
      if len(line) > 11:
        if line[0:11] == "data file: ":
          filename = line[11:]
          # make sure filename has full path if nhdr has fullpath
          filename = os.path.join(os.path.dirname(nhdr), filename)
          break
    return filename
  except IOError as e:
    # if this file cannot be opened, no way to find an associated file
    # just return empty string
    return ""

def remove_nrrd_files(nhdr):
  # first open the .nhdr file, look for the corresponding data file
  # if not found, only remove the given file, otherwise delete both
  # also remove the .completed file
  completedfile = nhdr + ".completed"
  errmsg = ""
  datafile = find_assoc_nrrd_data_file(nhdr)
  if datafile:
    errmsg += fm.delete_file(datafile)
  errmsg += fm.delete_file(nhdr)
  errmsg += fm.delete_file(completedfile)
  return errmsg

def get_type_from_nrrd(nhdr):
  """Return the type field from the nrrd header """
  fd = open(nhdr, 'r')
  nrrdtype = ""
  for line in fd:
    fields = line.split(':')
    if len(fields) >= 2 and fields[0] == "type":
      # found the line we are interested in.
      nrrdtype = fields[1].strip()
      break
  return nrrdtype

def get_space_directions_from_nrrd(nhdr):
  """ Return the space directions field from the nrrd header """
  fd = open(nhdr, 'r')
  space_dirs = []
  for line in fd:
    fields = line.split()
    if len(fields) > 2 and fields[0] == "space" and fields[1] == "directions:":
      # found the line we're interested in.  
      vecs = fields[2:]
      for vec in vecs:
        if vec != "none":
          vals = [ float(v) for v in vec.strip('(').strip(')').split(',') ]
          space_dirs.append(vals)
      # no need to keep reading lines
      break
  return space_dirs

def get_spacing_from_nrrd(nhdr):
  """Calculate the spacing in each direction from a 3-dimensional nrrd """
  fd = open(nhdr, 'r')
  spacing = []
  for line in fd:
    fields = line.split()
    if len(fields) > 2 and fields[0] == "space" and fields[1] == "directions:":
      # found the line we're interested in.  
      vecs = fields[2:]
      for vec in vecs:
        if vec != "none":
          vals = [ float(v) for v in vec.strip('(').strip(')').split(',') ]
          vals_squared = map(lambda x: x**2, vals)
          sum_vals_squared = reduce(lambda x,y: x+y, vals_squared)
          magnitude = math.sqrt(sum_vals_squared)
          spacing.append(magnitude)
      # no need to keep reading lines
      break
  return spacing

def get_origin_from_nrrd(nhdr):
  """Return the space origin field from the nrrd header """
  fd = open(nhdr, 'r')
  origin = []
  for line in fd:
    fields = line.split(':')
    if len(fields) >= 2 and fields[0] == "space origin":
      # found the line we are interested in.
      origin = fields[1].strip().strip('(').strip(')').split(',')
      origin = [float(x) for x in origin]
      break
  return origin

def get_sizes_from_nrrd(nhdr):
  """Return the sizes field from the nrrd header """
  fd = open(nhdr, 'r')
  sizes = []
  for line in fd:
    fields = line.split()
    if len(fields) >= 2 and fields[0] == "sizes:":
      # found the line we are interested in.
      sizes = fields[1:]
      sizes = [int(x) for x in sizes]
      break
  return sizes

def get_kinds_from_nrrd(nhdr):
  """Return the kinds field from the nrrd header """
  fd = open(nhdr, 'r')
  kinds=[]
  for line in fd:
    fields = line.split()
    if len(fields) > 2 and fields[0] == "kinds:":
      # found the line we are interested in.
      kinds = fields[1:]
      break
  return kinds

def get_data_file_from_nrrd(nhdr):
  """Return the data file field from the nrrd header """
  fd = open(nhdr, 'r')
  data_file = ""
  for line in fd:
    fields = line.split(':')
    print(fields)
    if len(fields) > 1 and fields[0] == "data file":
      # found the line we are interested in.
      data_file = fields[1].strip()
      break
  return data_file

class nrrdTransform():
  """ This class can modify NRRD headers.  These functions rely heavily on cat and sed."""
  def __init__(self):
    self.file = ""
    self.origin = [0,0,0]
    self.spacing = [ [1,0,0], [0,1,0], [0,0,1] ]
    self.space_directions = "(1,0,0), (0,1,0), (0,0,1)"
    self.kinds = "domain domain domain"
    self.space_dimension = None
    self.space = "left-posterior-superior"

  def to_2D_tensor(self):
    self.origin = [0,0]
    self.spacing = [ [1,0], [0,1] ]
    self.space_directions = "none (1,0) (0,1)"
    self.kinds = "vector domain domain"
    self.space_dimension = 2
    self.space = None
    [success, errmsg] = self.set_nrrd_origin()
    [s, e] = self.set_nrrd_space_directions()
    success = success and s
    errmsg += e
    [s, e] = self.set_nrrd_kinds()
    success = success and s
    errmsg += e
    [s, e] = self.add_nrrd_space_dimension()
    success = success and s
    errmsg += e
    [s, e] = self.remove_nrrd_space()
    success = success and s
    errmsg += e
    print("success?", success)
    print("errmsg: ", errmsg)

  def to_3D_tensor(self):
    self.origin = [0,0,0]
    self.spacing = [ [1,0,0], [0,1,0], [0,0,1] ]
    self.space_directions = "none (1,0,0) (0,1,0) (0,0,1)"
    self.kinds = "vector domain domain domain"
    self.space_dimension = 3
    self.space = None
    [success, errmsg] = self.set_nrrd_origin()
    [s, e] = self.set_nrrd_space_directions()
    success = success and s
    errmsg += e
    [s, e] = self.set_nrrd_kinds()
    success = success and s
    errmsg += e
    [s, e] = self.add_nrrd_space_dimension()
    success = success and s
    errmsg += e
    [s, e] = self.remove_nrrd_space()
    success = success and s
    errmsg += e
    print("success?", success)
    print("errmsg: ", errmsg)
    
  def set_nrrd_origin(self):
    """ Set the nrrd header origin to the values contained in self.origin """
    errmsg = ""
    success = False
    if self.file:
      fileroot, fileext = os.path.splitext(self.file)
      tempfile = fileroot + "_temp" + fileext
      cmds = []
      origin_str = "("
      for val in self.origin:
        origin_str += str(val) + ","
      origin_str = origin_str.strip(',')
      origin_str += ")"
      cmds.append('cat %s | sed s/"space origin.*"/"space origin: %s"/ > %s' % (self.file, origin_str, tempfile))
      cmds.append('cat %s > %s' % (tempfile, self.file))
      cmds.append('rm %s' % (tempfile))
      for cmd in cmds:
        try:
          os.system(cmd)
          success = True
        except OSError as e:
          errmsg = "Caught an error <%s> while running the command %s" % (str(e), cmd)
    return success, errmsg

  def set_nrrd_space_directions_from_spacing(self):
    """ Set the nrrd space directions to the values contained in self.spacing """
    errmsg = ""
    success = False
    if self.file:
      fileroot, fileext = os.path.splitext(self.file)
      tempfile = fileroot + "_temp" + fileext
      cmds = []
      spacing_str = ""
      for dim in self.spacing:
        spacing_str += " ("
        for val in dim:
          spacing_str += str(val) + ","
        spacing_str = spacing_str.strip(',')
        spacing_str += ")"
      cmds.append('cat %s | sed s/"space directions.*"/"space directions: %s"/ > %s' % (self.file, spacing_str, tempfile))
      cmds.append('cat %s > %s' % (tempfile, self.file))
      cmds.append('rm %s' % (tempfile))
      for cmd in cmds:
        try:
          os.system(cmd)
          success = True
        except OSError as e:
          errmsg = "Caught an error <%s> while running the command %s" % (str(e), cmd)
    return success, errmsg

  def set_nrrd_space_directions(self):
    """ Set the nrrd space directions to the values contained in self.space_directions """
    errmsg = ""
    success = False
    if self.file:
      fileroot, fileext = os.path.splitext(self.file)
      tempfile = fileroot + "_temp" + fileext
      cmds = []
      cmds.append('cat %s | sed s/"space directions.*"/"space directions: %s"/ > %s' % (self.file, self.space_directions, tempfile))
      cmds.append('cat %s > %s' % (tempfile, self.file))
      cmds.append('rm %s' % (tempfile))
      for cmd in cmds:
        try:
          os.system(cmd)
          success = True
        except OSError as e:
          errmsg = "Caught an error <%s> while running the command %s" % (str(e), cmd)
    return success, errmsg

  def set_nrrd_kinds(self):
    """ Set the nrrd header kinds to the values contained in self.kinds """
    errmsg = ""
    success = False
    if self.file:
      fileroot, fileext = os.path.splitext(self.file)
      tempfile = fileroot + "_temp" + fileext
      cmds = []
      cmds.append('cat %s | sed s/"kinds.*"/"kinds: %s"/ > %s' % (self.file, self.kinds, tempfile))
      cmds.append('cat %s > %s' % (tempfile, self.file))
      cmds.append('rm %s' % (tempfile))
      for cmd in cmds:
        try:
          os.system(cmd)
          success = True
        except OSError as e:
          errmsg = "Caught an error <%s> while running the command %s" % (str(e), cmd)
    return success, errmsg

  def add_nrrd_space_dimension(self):
    """ Add a nrrd header space dimension entry using the values contained in self.space_dimension """
    errmsg = ""
    success = False
    if self.file:
      fileroot, fileext = os.path.splitext(self.file)
      tempfile = fileroot + "_temp" + fileext
      cmds = []
      cmds.append('cat %s | sed 5s/"^//p; 5s/^.*/space dimension: %s"/ > %s' % (self.file, self.space_dimension, tempfile))
      cmds.append('cat %s > %s' % (tempfile, self.file))
      cmds.append('rm %s' % (tempfile))
      for cmd in cmds:
        try:
          os.system(cmd)
          success = True
        except OSError as e:
          errmsg = "Caught an error <%s> while running the command %s" % (str(e), cmd)
    return success, errmsg

  def remove_nrrd_space(self):
    """ Remove a nrrd header space entry """
    errmsg = ""
    success = False
    if self.file:
      fileroot, fileext = os.path.splitext(self.file)
      tempfile = fileroot + "_temp" + fileext
      cmds = []
      cmds.append('cat %s | sed /"^space:.*"/"d" > %s' % (self.file, tempfile))
      cmds.append('cat %s > %s' % (tempfile, self.file))
      cmds.append('rm %s' % (tempfile))
      for cmd in cmds:
        try:
          os.system(cmd)
          success = True
        except OSError as e:
          errmsg = "Caught an error <%s> while running the command %s" % (str(e), cmd)
    return success, errmsg

  def get_nrrd_data_type(self):
    """ Return the nrrd data type found in self.file """
    return get_type_from_nrrd(self.file)

  def get_nrrd_data_file(self):
    """ Return the nrrd header data file found in self.file """
    return get_data_file_from_nrrd(self.file)

  def set_nrrd_data_file(self):
    """ Set the nrrd header data file to the values contained in self.data_file """
    errmsg = ""
    success = False
    if self.file:
      fileroot, fileext = os.path.splitext(self.file)
      tempfile = fileroot + "_temp" + fileext
      cmds = []
      escaped_data_file = sed_escape_file(self.data_file)
      cmds.append('cat %s | sed s/"data file.*"/"data file: %s"/ > %s' % (self.file, escaped_data_file, tempfile))
      cmds.append('cat %s > %s' % (tempfile, self.file))
      cmds.append('rm %s' % (tempfile))
      for cmd in cmds:
        print(cmd)
        try:
          os.system(cmd)
          success = True
        except OSError as e:
          errmsg = "Caught an error <%s> while running the command %s" % (str(e), cmd)
    return success, errmsg
    
  def copy_hdr(self):
    """ Copy the header information from self.hdrToCopy into self.file """
    errmsg = ""
    success = False
    if self.file:
      cmds = []
      cmds.append('cat %s > %s' % (self.hdrToCopy, self.file))
      for cmd in cmds:
        print(cmd)
        try:
          os.system(cmd)
          success = True
        except OSError as e:
          errmsg = "Caught an error <%s> while running the command %s" % (str(e), cmd)
    return success, errmsg

  def write_gradients(self):
    """ Write the gradient information from self.file into self.gradientFile"""
    errmsg = ""
    success = False
    if self.file:
      cmds = []
      cmds.append("""cat %s | sed '/DWMRI_gradient.*/ '!'d' | sed s/"DWMRI_gradient_00.*:= "/""/ > %s""" % (self.file, self.gradientFile))
      for cmd in cmds:
        print(cmd)
        try:
          os.system(cmd)
          success = True
        except OSError as e:
          errmsg = "Caught an error <%s> while running the command %s" % (str(e), cmd)

    return success, errmsg
