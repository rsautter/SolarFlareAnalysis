from distutils.core import setup
from Cython.Build import cythonize
import numpy
import os
import sys

print("Numpy Location:"+numpy.get_include())
setup(ext_modules=cythonize(["Preprocessing/*.pyx","Metrics/*.pyx","Image/*.pyx"], annotate=True,
		compiler_directives={
			'optimize.use_switch': True,
			'initializedcheck':False,
			'overflowcheck':False,
			'optimize.unpack_method_calls':True,
			'boundscheck':False,
			'profile': False,
			'infer_types':True,
			'cdivision_warnings': False,
			'cdivision': True,
			'wraparound': False,
			'boundscheck': False},
                include_path=[numpy.get_include()]
),
include_dirs=[numpy.get_include()]
)
print("Moving HTML files....")
os.system("mv *.html html/ 2>/dev/null")
os.system("mv Metrics/*.html html/Metrics/ 2>/dev/null")
os.system("mv Image/*.html html/Image/ 2>/dev/null")
os.system("mv Preprocessing/*.html html/Preprocessing/ 2>/dev/null")
print("All done!")
