#!/usr/bin/env python
import os, sys

sys.path.insert(0, os.path.join(os.environ['PETSC_DIR'], 'config'))
sys.path.insert(0, os.path.join(os.environ['PETSC_DIR'], 'config', 'BuildSystem'))

import script

class ConfigReader(script.Script):
  def __init__(self):
    import RDict
    import os

    rootdir = os.getenv('SRMG_DIR', '.')
    self.archDir = os.path.join(rootdir, os.environ['PETSC_ARCH'])
    if not os.path.isdir(self.archDir):
      os.mkdir(self.archDir)
    argDB = RDict.RDict(None, None, 0, 0)
    argDB.saveFilename = os.path.join(os.environ['PETSC_DIR'], os.environ['PETSC_ARCH'], 'lib', 'petsc', 'conf', 'RDict.db')
    argDB.readonly = 1
    argDB.load()
    script.Script.__init__(self, argDB = argDB)
    return

  def run(self):
    self.setup()
    self.framework = self.loadConfigure()
    return

class GMakefileGenerator(ConfigReader):
  def __init__(self):
    ConfigReader.__init__(self)
    return

  def run(self):
    ConfigReader.run(self)
    with file(os.path.join(self.archDir, 'gmakefile'), 'w') as f:
      self.framework.outputMakeMacro(f, 'PYTHON', sys.executable)

      self.framework.outputMakeMacro(f, 'PETSC_DIR',  os.environ['PETSC_DIR'])
      self.framework.outputMakeMacro(f, 'PETSC_ARCH', os.environ['PETSC_ARCH'])
      f.write('include ${PETSC_DIR}/lib/petsc/conf/variables\n\n')

      f.write('include ../base.mk')
    return

if __name__ == '__main__':
  GMakefileGenerator().run()
