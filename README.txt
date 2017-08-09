MLSandbox has the following dependencies:

    boost-python
    numpy
    gsl
    BoostNumpy*

*BoostNumpy can be installed from https://github.com/martwo/BoostNumpy. Clone the git repo and follow the instructions in the README. It is recommended to install the library in /usr.

MLSandbox have to be build in a separate build directory. To do so create a build directory and step into it and run `cmake path/to/MLSandbox-source`. To load the MLSandbox enviroment, and thus get easy access to the pybindings, the env_shell.sh needs to be exectuted.

The signal subtration study scripts in the signalsubtraction folder depends on the following python modules:
    numpy
    scipy
    healpy
    matplotlib
    pickle
    dashi*

*dashi is a python module that wraps around the numpy histogram to give it similar functionalities of a root histogram. dashi can be installed by the following command `pip install https://github.com/emiddell/dashi/zipball/master`


