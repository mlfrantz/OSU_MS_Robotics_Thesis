from setuptools import setup

setup(name='sas_utils',
      version='0.0.1',
      description='Wrapper Package for NSF S&AS Base Classes',
      url='http://tb.d',
      author='Seth McCammon',
      author_email='mccammos@oregonstate.edu',
      license='None',
      packages=['sas_utils'],
      install_requires=['numpy>=1.14.0', 'GPy>=1.8.5', 'matplotlib>=2.1.2', 'haversine>=0.4.5', 'scipy>=1.0.0', 'deepdish>=0.3.6', 'shapely>=1.6.4.post2'],
      zip_safe=False)