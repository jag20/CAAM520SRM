#! /usr/bin/env python
import os, sys, math, numpy as np
import re
sys.path.insert(0, os.path.join(os.environ['PETSC_DIR'], 'config'))
sys.path.insert(0, os.path.join(os.environ['PETSC_DIR'], 'config', 'BuildSystem'))

import subprocess
import numpy as np
import scipy
import matplotlib.pyplot as plt

import script
class ConfigReader(script.Script):
  def __init__(self):
    import RDict
    import os

    argDB = RDict.RDict(None, None, 0, 0)
    argDB.saveFilename = os.path.join(os.environ['PETSC_DIR'], os.environ['PETSC_ARCH'], 'lib','petsc','conf', 'RDict.db')
    argDB.load()
    script.Script.__init__(self, argDB = argDB)
    return

  def get(self, key):
    return self.framework.argDB[key]

  def getModule(self, modname):
    return self.framework.require(modname, None)

  def run(self):
    self.setup()
    self.framework = self.loadConfigure()

config = ConfigReader()
config.run()
sharedExt = config.getModule('config.setCompilers').sharedLibraryExt


options = ['-snes_type', 'srmg', '-dll_prepend', '$SRMG_DIR/$PETSC_ARCH/lib/libsrmg.'+sharedExt, '-log_view', '-MMS', '2' ,'-buffer_size', '1']
cmd = '$PETSC_DIR/src/snes/examples/tutorials/ex5 '+' '.join(options)

from pylab import plot, loglog, show, title, xlabel, ylabel, legend, text, figure
n = 3
rmax = 2 #refinement ratio
rs = range(1,rmax+1)
for r in rs:
    print 'refinement ratio: %d' %r
    Ns = []
    times = []
    for da_grid in [ 2**j for j in range(1,n+1) ]:
        N = da_grid**2
        options = [' -snes_srmg_refinement_ratio', str(r),' -snes_srmg_levels', str(int(math.log(N,4))), '-da_grid_x', str(da_grid), '-da_grid_y', str(da_grid)]
        cmd = cmd + ' '.join(options)
        out = subprocess.check_output(cmd + " | grep \"^Time (sec):\"", shell=True)
        time = out.split()[2]
        print 'N: %s time: %ss' %(N,time)
        times.append(float(time))
        Ns.append(N)
    slope = np.polyfit(np.log(times),np.log(Ns),1)[0]
    print 'log(time) vs log(N) linear fit slope: %e' %slope

    ##########  Plots  ####################
    f = figure()
    p = f.add_subplot(111)
    loglog(Ns, times,marker='o',linestyle='--')
    title('SNES with SRMG: ex5 MMS2 runtime with refinement ratio '+str(r))
    xlabel('N')
    ylabel('Time(s)')
    text(0.1, 0.9,'log-log linear fit slope: '+str(slope), ha='left', va='top', transform = p.transAxes)
show()