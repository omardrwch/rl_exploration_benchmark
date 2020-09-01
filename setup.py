from setuptools import setup, find_packages

packages_ = find_packages()
packages = [p for p in packages_ if not(p == 'tests')]

setup(name='rlxp',
      version='0.0.1-dev',
      description='Simple environments to benchmark exploration in reinforcement learning algorithms',
      url='https://github.com/omardrwch/rl_exploration_benchmark',
      author='Omar Darwiche Domingues',
      author_email='',
      license='MIT',
      packages=packages,
      install_requires=['numpy', 'gym', 'pytest', 'PyOpenGL', 'PyOpenGL_accelerate', 'pygame'],
      zip_safe=False)