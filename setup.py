from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ext_modules = [
    Extension(
        'lpcnet',
        ["lpcnet.pyx"],
        # extra_compile_args=['-mavx2', '-mfma', '-O3', '-g']
        extra_compile_args=['-O3', '-g'] # if cpu report "illegal instruction"
    )
]

setup(
    name='pylpcnet',
    version='1.2-biaobei', # 1.0 raw, 1.1.0 aishellC0896, 1.1.1 aishellC0002, 1.2 biaobei
    ext_modules=cythonize(ext_modules)
)
