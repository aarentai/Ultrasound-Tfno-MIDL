# tools for managing files, moving, removing etc.
import os

def delete_file(file):
  errmsg = ""
  try:
    os.remove(file)
  except OSError as e:
    errmsg += "Error removing file <%s>: %s/n" % (file, e)
  return errmsg
  
def create_dir_hierarchy(path):
    # create all directories that do not exist in path
    # let exception be thrown if/when user does not have write permission
    dirname = os.path.dirname(path)
    subdirs = []
    subdirs.append(dirname)
    vals=os.path.split(dirname)

    while vals[1]:
        subdirs.append(vals[0])
        vals=os.path.split(vals[0])

    for elem in reversed(subdirs):
        if not os.path.exists(elem):
            os.mkdir(elem)
