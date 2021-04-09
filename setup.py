from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

with open("gribflow/version.py") as f:
    exec(f.read())

extra_setuptools_args = dict(
    tests_require=['pytest']
)

setup(
    name ='gribflow',
    version = __version__,
    author ='Stan Biryukov',
    author_email ='stanley.biryukov@tomorrow.io',
    url = 'git@github.com:stanbiryukov/gribflow.git',
    install_requires = [requirements,],
    package_data = {'gribflow':['resources/*']},
    packages = find_packages(exclude=['gribflow/tests']),
    license = 'MIT',
    description='gribflow: Read NOAA gribs',
    long_description= "gribflow makes NOAA model files usable for any downstream process",
    keywords = ['optical-flow', 'computer-vision', 'grib', 'NOAA', 'data', 'weather'],
    classifiers = [
        "Programming Language :: Python :: 3.7",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License"
    ],
    **extra_setuptools_args
)