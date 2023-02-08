from setuptools import setup, find_packages

setup(name='limbus-components',
      version='0.1.0.dev',
      description='Components to be used with limbus.',
      long_description=open('README.md').read(),
      long_description_content_type='text/markdown',
      author='Luis Ferraz',
      url='https://github.com/kornia/limbus-components',
      install_requires=[
          'torch',
          'numpy',
          'kornia',
          'opencv-python<4.7'  # 4.7 doesn't work in my computer
      ],
      extras_require={
          'dev': [
              'limbus',
              'pytest',
              'pytest-flake8',
              'pytest-cov',
              'pytest-mypy',
              'pytest-pydocstyle',
              'pytest-asyncio',
              'mypy',  # TODO: check if we can remove the deps without pytest-*
              'pydocstyle',
              'flake8<5.0.0',  # last versions of flake8 are not compatible with pytest-flake8==1.1.1 (lastest version)
              'pep8-naming',
          ]
      },
      packages=find_packages(where='.'),
      package_dir={'': '.'},
      package_data={'': ['*.yml']},
      include_package_data=True
      )
