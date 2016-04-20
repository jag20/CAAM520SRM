#! /usr/bin/env python
import scipy
import matplotlib.pyplot as plt
import os
import subprocess
import numpy as np
from pylab import legend, plot, loglog, show, title, xlabel, ylabel, figure, savefig

mms = [1,2,3,4]
for mms_val in mms:
    print("MMS: ", mms_val)
    sizes = []
    errors = []
    for k in range(4):
        Nx = 10*2**k
        modname = 'perf%d' %k
        options = ['-mms', str(mms_val), '-da_grid_x', str(Nx), '-da_grid_y', str(Nx), '-log_view', ':%s.py:ascii_info_detail' %modname] 
        path =  '.' + os.environ['PETSC_DIR'] + '/src/snes/examples/tutorials/ex5 '
        output = subprocess.check_output('./ex5 ' +' '.join(options), shell=True)
        start_index = output.find('l2')
        if start_index == -1:
            raise Exception ("l2 norm Not found")
        else:
            print(output)
            if mms_val < 3:
                errors.append(float(output[start_index+3:start_index+14]))
            else:
                errors.append(float(output[start_index+3:start_index+13]))
        #perfmod = __import__(modname)
        sizes.append(Nx**2)


########### Linear Fit ################
    print zip(sizes, errors)
    sizes = np.array(sizes)

    x = np.log10(np.array(sizes))
    y = np.log10(np.array(errors))
    X = np.hstack((np.ones((x.shape[0],1)),x.reshape((x.shape[0],1))))

    beta = np.dot(np.linalg.pinv(np.dot(X.transpose(),X)),X.transpose())
    beta = np.dot(beta,y.reshape((y.shape[0],1)))

    print("y-intercept: ", beta[0][0])
    print("slope: ", beta[1][0])


##########  Plots  ####################
    plot(sizes, errors)
    title('SNES ex5')
    xlabel('Problem Size $N$')
    ylabel('Error')
    savefig('SNES ex5_' + str(mms_val)+'.png')
    figure()
    loglog(sizes, errors, sizes, 0.9*sizes**-1.5)
    title('loglog SNES ex5')
    xlabel('Problem Size $N$')
    ylabel('Error')
    savefig('loglog SNES ex5_'+str(mms_val)+'.png')
    #show()
