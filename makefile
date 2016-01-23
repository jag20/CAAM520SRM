all : $(PETSC_ARCH)/gmakefile
	$(MAKE) -C $(PETSC_ARCH) -f gmakefile
	@echo "Build complete in $(PETSC_ARCH).  Use make test to test."

$(PETSC_ARCH)/gmakefile: ./config/gmakegen.py
	$(PYTHON) ./config/gmakegen.py

test : all
	$(MAKE) -C $(PETSC_ARCH) test

clean :
	$(MAKE) -C $(PETSC_ARCH) clean

.PHONY: all test clean

include $(PETSC_DIR)/$(PETSC_ARCH)/lib/petsc/conf/petscvariables
