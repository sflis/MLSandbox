# MLSandbox
A small framework for binned maximum likelihood analyses using mixture models (signal+background), i.e only fitting the signal fraction. An implementation of the signal subtracted likelihood is available as one of the likelihood formulations

The main code base is in C++ but this project is meant to be used through its python bindings which exposes all the needed functunality to perform most analyses.

Dependencies
------------
To build MLSandbox the following dependencies are needed: gsl, boost, numpy and BoostNumpy (https://github.com/martwo/BoostNumpy)

Building MLSandbox
------------------

At the moment only manual build is possible. The easiest way to build the project is to create a build directory, step into the build directory and execute:
```shell
  cmake path/to/MLSandbox/source
  make
```
Besides building the C++ library and the python module this will also create a env_shell.sh. Running this file with `sh` will create an enviroment where the MLSandbox python module is in the `PYTHON_PATH`. Therefore after running:
```shell
  sh env_shell.sh
```

you should be able to import MLSandbox in your python script simply with
```python
import MLSandbox
```

Status
------
Status of build: 

![Test Status Travis-CI](https://travis-ci.org/sflis/MLSandbox.svg?branch=master)
  

