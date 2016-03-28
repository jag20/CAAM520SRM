#! /usr/bin/env python
import subprocess
import numpy as np
import scipy
import matplotlib.pyplot as plt


Ns = []
errors = []

k = 6

options = ['-snes_type', 'srmg', '-dll_prepend', '$SRMG_DIR/$PETSC_ARCH/lib/libsrmg.so', '-MMS', '2', '-snes_srmg_levels', str(k) ,'-buffer_size', '0', '-snes_srmg_refinement_ratio', '2']
#options = ['-snes_type', 'srmg', '-dll_prepend', '$SRMG_DIR/$PETSC_ARCH/lib/libsrmg.so', '-MMS', '2', '-snes_srmg_levels', str(k),'-buffer_size', '1', '-da_grid_x 7', '-da_grid_y 7']

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

