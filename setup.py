from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy as np
import sys
sys.setrecursionlimit(1000000)

with open('requirements.txt') as f:
    requirements = f.readlines()
  
extensions = [
    Extension(
        "REVEALER.REVEALER_Cython",
        ["REVEALER/REVEALER_Cython.pyx"],
        include_dirs=[np.get_include()]
    ),
    Extension(
        "REVEALER.CheckGrid",
        ["REVEALER/CheckGrid.pyx"],
        include_dirs=[np.get_include()]
    ),
    Extension(
        "REVEALER.MutMaker",
        ["REVEALER/MutMaker.pyx"],
        include_dirs=[np.get_include()]
    ),
    Extension(
        "REVEALER.REVEALER_runbenchmark",
        ["REVEALER/REVEALER_runbenchmark.pyx"],
        include_dirs=[np.get_include()]
    )
]

long_description = '#TODO'
  
setup(
        name ='REVEALER',
        version ='2.0.2',
        author="Jiayan(Yoshii) Ma",
        author_email="jim095@ucsd.edu",
        url ='https://github.com/yoshihiko1218/REVEALER',
        description="REVEALER#TODO",
        long_description = long_description,
        long_description_content_type ="text/markdown",
        license ='MIT',
        packages = find_packages(),
        ext_modules=cythonize(extensions,language_level = "3"),
        cmdclass={'build_ext': build_ext},
        entry_points ={
            'console_scripts': [
                'REVEALER_preprocess = REVEALER.REVEALER_preprocess:main',
                'REVEALER = REVEALER.REVEALER:main'
            ]
        },
        classifiers =[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent"
        ],
        keywords ='REVEALER',
        install_requires = requirements,
        zip_safe = False,
        include_package_data=True
)
