#! /usr/bin/env python
import pickle
import numpy as np
import scipy
import matplotlib.pyplot as plt
from pylab import legend, plot, loglog, show, title, xlabel, ylabel, figure

k = 6  # how many levels to use
mms = 2 # which mms
refine_rat = 2 # refinement ratio
x_size = 25
y_size = 25

# goes up the patches in the output and parses to Ns and errors
def parseErrors(levels,out):
    Ns = []
    errors = []
    for L in range(1,levels+1):
        lookup_string = 'L: ' + str(L) + ' N: '
        start_index = out.rfind('L: ' + str(L) + ' N: ')
        N_end_index = out.find(' ',start_index+ len(lookup_string))
        N = out[start_index+ len(lookup_string):N_end_index]
        error_start_index = out.find('l2 ',N_end_index)
        error_end_index = out.find(' ',error_start_index+3)
        error = out[error_start_index+3:error_end_index]
        errors.append(float(error))
        Ns.append(int(N))
    return (Ns,errors)



for buffsize in [0, 1, 2]:
    # same file name as in gen_patch_data.py
    file_Name = "data/patchlvls_" + str(k) +"_x_" +str(x_size) +"_y_"+str(y_size) + "_buffer_" + str(buffsize) + "_mms_" + str(mms) + "_refrat_" + str(refine_rat)
    fileObject = open(file_Name,'r')
    out = fileObject.read()
    (Ns,errors) = parseErrors(k,out)
    #figure(1)
    #plot(Ns, errors)
    figure(2)
    loglog(Ns, errors)
    
##########  Plots  ####################




figure(2)
title('Patch Convergence for Varying Buffer Sizes')
xlabel('log N')
ylabel('log Error')

legend([0,1,2])

show()
