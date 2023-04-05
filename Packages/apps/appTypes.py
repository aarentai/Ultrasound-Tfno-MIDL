import os
import subprocess
import logging
import time

from util import monitor as mon
from util import YAMLcfg
from data import nrrd

nested_indent="   "

class BasicApp:
  def __init__(self, function_to_run=None, config=None):
    self.name = "Basic"
    self.desc = """ A basic app """
    self.timer = mon.Timer(False)
    self.run_function = function_to_run
    if config:
      self.set_config(config)

  def run(self):
    self.timer.reset()
    
    if self.run_function:
      self.run_function()
    else:
      print("No function provided to", self.name)
      
    self.timer.pause()

  def set_config(self, config):
    self.cfg = config

  def get_run_time(self):
    return self.timer.timesofar()
# end class BasicApp

CmdLineAppConfigSpec = {
  'inputPrefix': YAMLcfg.Param(default = './',
                               comment = 'Prefix where input data can be found.'),
  'outputPrefix': YAMLcfg.Param(default = './',
                                comment = 'Prefix where output data should be written.'),
  'commandPrefix': YAMLcfg.Param(default = '',
                               comment = 'Prefix where commands can be found.')
  }
    
class CmdLineApp(BasicApp):
  def __init__(self, cmds, files_to_clean, config=None):
    super().__init__(config = config)
    self.name = "Command Line"
    self.desc = """An app that runs the system commands
%s
and always needs to be rerun.""" % (cmds)
    self.commands = cmds
    self.cwd = './'
    self.files_to_clean = files_to_clean
    self.process = None

  def __str__(self, level=0):
    indent = nested_indent * level
    msg = "%sExecutes commands:\n%s   %s\n" % (indent, indent, str(self.commands))
    msg += "%sRemoves files:\n" % (indent)
    for file in self.files_to_clean:
      msg += "%s   %s\n" % (indent, file)
    return msg

  # cannot resume processing currently, so this app is never considered finished or started
  # meaning it should always be rerun
  def is_completed(self, version):
    return False

  def is_started(self, version):
    return False

  def prepare_for_restart(self):
    return ""

  def cleanup(self):
    """Remove any temporary files that are not needed after this stage"""
    errmsg = ""
    for file in self.files_to_clean:
      try:
        errmsg += nrrd.remove_nrrd_files(file)
      except OSError as e:
        errmsg += "Error cleaning up file <%s>: %s/n" % (file, e)
    return errmsg

  def run(self):
    self.timer.reset()
    success = True
    alloutdata = []
    if self.commands:
      for cmd in self.commands:
        try:
          self.process = subprocess.Popen(cmd, shell=True, cwd=self.cwd,
                                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT) 
        except OSError as e:
          errmsg = "Caught an error <%s> starting the subprocess %s" % (str(e), cmd)
          success = False
          self.timer.pause()
          return success, errmsg
          
        outdata, errdata = self.process.communicate() # errdata should be empty since stderr redirected to stdout
        alloutdata.append(outdata)
        success = success and not self.process.returncode
    else:
      errmsg = "No commands provided"
      success = False
      self.timer.pause()
      return success, errmsg
    self.timer.pause()
    return success, alloutdata
    
  def stop(self):
    # stop really means restart this stage, so just kill any existing processes
    if self.process:
      self.process.terminate() # try to end nicely first
      self.process.kill() # make sure it's really ended
    self.timer.pause()

  #def get_run_time(self):
  #  return self.timer.timesofar()
# end class CmdLineApp
