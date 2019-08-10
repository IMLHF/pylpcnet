from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ext_modules = [
    Extension(
        'lpcnet',
        ["lpcnet.pyx"],
        extra_compile_args=['-mavx2', '-mfma', '-O3', '-g']
    )
]

setup(
    name='pylpcnet',
    ext_modules=cythonize(ext_modules)
)
