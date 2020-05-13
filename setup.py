import sys

from setuptools import setup

if sys.version_info < (3, 5):
    print("Python 3.5 or higher required, please upgrade.")
    sys.exit(1)

version = "0.1"

setup(name="firedrake_ts",
      description="PETSc TS + Firedrake",
      version=version,
      author="Ivan Yashchuk",
      license="MIT",
      packages=["firedrake_ts"],)
