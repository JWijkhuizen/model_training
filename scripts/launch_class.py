#!/usr/bin/env python
import roslaunch

class launch_class:
  def __init__(self, package, names, files):
    self.package = package
    self.names = names
    self.file = dict()
    self.launchfile = dict()
    self.args = dict()
    self.uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    roslaunch.configure_logging(self.uuid)

    # Fill the launchfiles and args
    for i in range(len(files)):
      self.add_file(names[i],files[i])

  def add_file(self, name, file):
    self.names.append(name)
    self.file[name] = roslaunch.rlutil.resolve_launch_arguments([self.package,file[0]])[0]
    if len(file) > 2:
      self.args[name] = file[1:]
      self.launchfile[name] = (self.file[name], self.args[name])
    elif len(file) == 1:
      self.args[name] = []
      self.launchfile[name] = self.file[name]
    elif len(file) == 2:
      self.args[name] = file[1]
      self.launchfile[name] = (self.file[name], self.args[name])

  def run(self,name):
  	return roslaunch.parent.ROSLaunchParent(self.uuid, [self.launchfile[name]])