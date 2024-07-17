from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy as np
import sys
sys.setrecursionlimit(1000000)

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

extensions = [
    Extension(
        "revealer.REVEALER_Cython",
        ["src/revealer/REVEALER_Cython.pyx"],
        include_dirs=[np.get_include()]
    ),
    Extension(
        "revealer.CheckGrid",
        ["src/revealer/CheckGrid.pyx"],
        include_dirs=[np.get_include()]
    ),
    Extension(
        "revealer.MutMaker",
        ["src/revealer/MutMaker.pyx"],
        include_dirs=[np.get_include()]
    ),
    Extension(
        "revealer.REVEALER_runbenchmark",
        ["src/revealer/REVEALER_runbenchmark.pyx"],
        include_dirs=[np.get_include()]
    )
]

long_description = '#TODO'

setup(
    name='REVEALER',
    version='2.1.6',
    author="Jiayan(Yoshii) Ma",
    author_email="jim095@ucsd.edu",
    url='https://github.com/yoshihiko1218/REVEALER',
    description="REVEALER#TODO",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='MIT',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    ext_modules=cythonize(extensions, language_level="3"),
    cmdclass={'build_ext': build_ext},
    entry_points={
        'console_scripts': [
            'REVEALER_preprocess = revealer.REVEALER_preprocess:main',
            'REVEALER = revealer.revealer:main',
            'REVEALER_test = revealer.REVEALER_test:main'
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    keywords='REVEALER',
    install_requires=requirements,
    python_requires='>=3.7, <3.10',
    zip_safe=False,
    include_package_data=True,
    package_data={
        '': ['src/revealer/data/*'],
    },
)

