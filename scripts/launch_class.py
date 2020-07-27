#!/usr/bin/env python
import roslaunch

class launch_class:
  def __init__(self, package, names, files):
    self.package = package
    self.names = names
    self.file = dict()
    self.launchfile = dict()
    self.args = dict()

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

  def new_uuid(self):
    uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    roslaunch.configure_logging(uuid)
    return uuid


  def run(self,name):
    # New uuid for every launch instance
    # I suspect that sometimes gazebo crashes if the same uuid is used again for many times.
    uuid = self.new_uuid()
    return roslaunch.parent.ROSLaunchParent(uuid, [self.launchfile[name]])