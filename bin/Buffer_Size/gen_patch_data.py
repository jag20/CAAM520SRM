#! /usr/bin/env python
import pickle
import subprocess

k = 6  # how many levels to use
mms = 2 # which mms
refine_rat = 2 # refinement ratio
x_size = 25
y_size = 25

for buffsize in [0, 1, 2]:
    options = ['-snes_type', 'srmg', '-dll_prepend', '$SRMG_DIR/$PETSC_ARCH/lib/libsrmg.so', '-mms', str(mms), '-snes_srmg_levels', str(k) ,'-buffer_size', str(buffsize), '-snes_srmg_refinement_ratio', str(refine_rat), '-da_grid_x',str(x_size),'-da_grid_y',str(y_size)]

    cmd = '$PETSC_DIR/src/snes/examples/tutorials/ex5 '+' '.join(options)
    out = subprocess.check_output(cmd, shell=True) 
    file_Name = "data/patchlvls_" + str(k) +"_x_" +str(x_size) +"_y_"+str(y_size) + "_buffer_" + str(buffsize) + "_mms_" + str(mms) + "_refrat_" + str(refine_rat)
    fileObject = open(file_Name,'wb')
    fileObject.write(out)
    fileObject.close()
 
