message(STATUS "Installing python package \"ndhist\" via setuptools.")

execute_process(
    COMMAND /usr/bin/env bash /home/sflis/i3/projects/MLSandbox/MLSandboxSrc/cmake/invoke_setup.py.sh  /home/sflis/i3/projects/MLSandbox/MLSandboxSrc /usr/local
    WORKING_DIRECTORY /home/sflis/i3/projects/MLSandbox/MLSandboxSrc/python
    ERROR_VARIABLE _SETUPTOOLS_ERR
    OUTPUT_VARIABLE _SETUPTOOLS_OUT
)

message(STATUS "${_SETUPTOOLS_OUT}")
if(${_SETUPTOOLS_ERR})
    message(STATUS "ERROR: Setuptools error output: ${_SETUPTOOLS_ERR}")
endif()
