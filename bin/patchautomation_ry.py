#! /usr/bin/python
import os, sys
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
Ns = []
errors = []

k = 6

options = ['-snes_type', 'srmg', '-dll_prepend', '$SRMG_DIR/$PETSC_ARCH/lib/libsrmg.'+sharedExt, '-MMS', '2', '-snes_srmg_levels', str(k) ,'-buffer_size', '0', '-snes_srmg_refinement_ratio', '2', '-snes_srmg_interp_order', '1' ]
#options = ['-snes_type', 'srmg', '-dll_prepend', '$SRMG_DIR/$PETSC_ARCH/lib/libsrmg.'+sharedExt, '-MMS', '2', '-snes_srmg_levels', str(k),'-buffer_size', '1', '-da_grid_x 7', '-da_grid_y 7']

cmd = '$PETSC_DIR/src/snes/examples/tutorials/ex5 '+' '.join(options)
out = subprocess.check_output(cmd, shell=True) 
print(out)

for L in range(1,k+1):
  lookup_string = 'L: ' + str(L) + ' N: '
  start_index = out.rfind('L: ' + str(L) + ' N: ')
  N_end_index = out.find(' ',start_index+ len(lookup_string))
  N = out[start_index+ len(lookup_string):N_end_index]
  print('N: '+ N)
  error_start_index = out.find('l2 ',N_end_index)
  error_end_index = out.find(' ',error_start_index+3)
  error = out[error_start_index+3:error_end_index]
  print('error:' + error)
  errors.append(float(error))
  Ns.append(int(N))



##########  Plots  ####################
from pylab import legend, plot, loglog, show, title, xlabel, ylabel, figure
plot(Ns, errors)
title('SNES ex5')
xlabel('N')
ylabel('Error')


figure()
loglog(Ns, errors)
title('loglog SNES ex5')
xlabel('N')
ylabel('Error')

show()

