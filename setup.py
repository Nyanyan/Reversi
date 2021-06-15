from distutils.core import setup, Extension
from Cython.Build import cythonize
from numpy import get_include # cimport numpy を使うため

filename = input()

ext = Extension("ai_cython_exe", sources=[filename], include_dirs=['.', get_include()])
setup(name="ai_cython_exe", ext_modules=cythonize([ext]))

f = open('ai_cython_exe.cp38-win_amd64.pyd')
f.close()
