# setup.py
from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        name="cython_modules.text_processing",
        sources=["cython_modules/text_processing.pyx"],
    )
]

setup(
    ext_modules=cythonize(extensions),
)
