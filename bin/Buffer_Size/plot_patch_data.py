#! /usr/bin/env python
import numpy as np
import scipy, math
import matplotlib.pyplot as plt
from pylab import legend, plot, loglog, show, title, xlabel, ylabel, figure

k = 3  # how many levels to use
mms = 2 # which mms
refine_rat = 2 # refinement ratio
x_size = 128
y_size = 128
buffSizes = [0,2,4,8,16]
# goes up the patches in the output and parses to Ns and errors
def parseErrors(levels,out):
    Ns = []
    errors = []
    for L in range(1,levels+1):
        lookup_string = 'L: ' + str(L) + ' N: '
        error         = 0.0
        for line in out.split('\n'):
            if line.find(lookup_string) >= 0:
                N      = int(line.split(' ')[3])
                error += float(line.split(' ')[8])
        errors.append(error)
        Ns.append(N)
    return (np.array(Ns),np.array(errors))


for buffsize in buffSizes:
    # same file name as in gen_patch_data.py
    file_Name = "data/patchlvls_" + str(k) +"_x_" +str(x_size) +"_y_"+str(y_size) + "_buffer_" + str(buffsize) + "_mms_" + str(mms) + "_refrat_" + str(refine_rat)
    fileObject = open(file_Name,'r')
    out = fileObject.read()
    (Ns,errors) = parseErrors(k,out)
    #figure(1)
    #plot(Ns, errors)
    figure(2)
    loglog(Ns, errors)
    # estimate slope
    slope = (math.log(errors[-1]) - math.log(errors[-2]))/(math.log(Ns[-1]) - math.log(Ns[-2]))
    print 'buffer:',buffsize,'slope:',slope
    
##########  Plots  ####################



loglog(Ns, 1000*Ns**-1.5)
figure(2)
title('Patch Convergence for Varying Buffer Sizes')
xlabel('log N')
ylabel('log Error')
legend(buffSizes)

show()
