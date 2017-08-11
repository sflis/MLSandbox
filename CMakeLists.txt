cmake_minimum_required (VERSION 2.8)
if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
  message(FATAL_ERROR "
***************************************************************************
***      In place build is a bad idea. Please build out-of-place.       ***
***   Run cmake /path/to/source from a directory != /path/to/source.    ***
***                                                                     ***
*** DON'T FORGET TO REMOVE THE CMakeCache.txt IN YOUR SOURCE DIRECTORY  ***
***                  THAT WAS JUST CREATED.                             ***
***************************************************************************
")
endif("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")


project(MLSandbox)

message(STATUS "Configuring project \"${PROJECT_NAME}\"")
message(STATUS "+    project source dir: ${PROJECT_SOURCE_DIR}")

set(MLSANDBOX_CONFIGURED_SUCCESSFULLY TRUE
    CACHE BOOL "${PROJECT_NAME} configured successfully." FORCE)

include(${CMAKE_SOURCE_DIR}/cmake/config.cmake)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    CACHE PATH "Output directory for shared libraries." FORCE)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    CACHE PATH "Output directory for executables like tests.")

include(${CMAKE_SOURCE_DIR}/cmake/add_python_module.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/find_tool.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/tools/boost.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/tools/python.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/tools/numpy.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/tools/boostnumpy.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/tools/FindGSL.cmake)
#include(${CMAKE_SOURCE_DIR}/cmake/tools/sphinx.cmake)

find_boost()
find_python()
find_numpy()
find_boostnumpy()
find_gsl()

set(${PROJECT_NAME}_CONFIGURED_SUCCESSFULLY TRUE)


if(NOT (GSL_FOUND AND BOOSTNUMPY_FOUND) AND DEFINED ENV{I3_PORTS})
      message("GSL and/or boost-numpy not found in usual location looking for I3_PORTS and cvmfs") 
      foreach(v 1.8 1.14 1.15 1.16)
        if(IS_DIRECTORY $ENV{I3_PORTS}/include/gsl-${v})
          set(GSL_VERSION "${v}")
          set(GSL_SUBDIR "gsl-${v}")
        endif(IS_DIRECTORY $ENV{I3_PORTS}/include/gsl-${v})
      endforeach(v 1.8 1.14 1.15 1.16)
      set(GSL_FOUND ON)
      set(GSL_INCLUDE_DIRS $ENV{I3_PORTS}/include/${GSL_SUBDIR})
      set(GSL_LIBRARIES $ENV{I3_PORTS}/lib/${GSL_SUBDIR}/libgsl.so)
      set(GSL_LIBRARIES $ENV{I3_PORTS}/lib/${GSL_SUBDIR}/libgsl.so $ENV{I3_PORTS}/lib/${GSL_SUBDIR}/libgslcblas.so)
      message("GSL FOUND")
      message("gsl path " ${GSL_INCLUDE_DIRS})
      
      set(BOOSTNUMPY_LIBRARIES $ENV{SROOT}/lib/libboost_numpy.so)# CACHE FILEPATH "boost::numpy library")
      set(BOOSTNUMPY_INCLUDE_DIRS $ENV{SROOT}/include)# CACHE PATH "Path to boost::numpy include dir")
      set(BOOSTNUMPY_FOUND TRUE)# "boost::numpy found")
      message("bn include" ${BOOSTNUMPY_INCLUDE_DIR})
      # boost::numpy can't work without the Numpy C API
      LIST(APPEND BOOSTNUMPY_INCLUDE_DIR ${NUMPY_INCLUDE_DIR})
      LIST(APPEND BOOSTNUMPY_LIBRARIES ${NUMPY_LIBRARIES})
      
    
endif()
if(NOT (BOOST_FOUND AND PYTHON_FOUND AND NUMPY_FOUND AND BOOSTNUMPY_FOUND AND GSL_FOUND))
    message("" ${BOOSTNUMPY_FOUND} "  " ${GSL_FOUND})
    set(${PROJECT_NAME}_CONFIGURED_SUCCESSFULLY FALSE)
endif()

if(${PROJECT_NAME}_CONFIGURED_SUCCESSFULLY)
    #set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)
    list(APPEND ${PROJECT_NAME}_INCLUDE_DIRS
        ${PYTHON_INCLUDE_DIRS}
        ${NUMPY_INCLUDE_DIRS}
        ${BOOST_INCLUDE_DIRS}
                ${BOOSTNUMPY_INCLUDE_DIRS}
        ${GSL_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/public/
        ${PROJECT_SOURCE_DIR}/private/
        
    )

    SET(LIBRARY_SRC
            private/MLSandbox/Distribution.cxx
            private/MLSandbox/Likelihood.cxx
            private/MLSandbox/Minimizer.cxx
            private/MLSandbox/FeldmanCousins.cxx
            private/MLSandbox/FCRanks.cxx
            private/MLSandbox/CombinedLikelihood.cxx
            private/MLSandbox/NeymanAnalysis.cxx
            private/MLSandbox/SignalContaminatedLH.cxx
            private/MLSandbox/LikelihoodCollection.cxx
    )

    include_directories(${${PROJECT_NAME}_INCLUDE_DIRS})
    add_library(mlsandbox ${LIBRARY_SRC})

    target_link_libraries(mlsandbox
        ${PYTHON_LIBRARIES}
        ${BOOST_LIBRARIES}
        ${BOOSTNUMPY_LIBRARIES}
        ${GSL_LIBRARIES}
    )

# Copy some cmake files over to the build directory.
    add_custom_command(TARGET mlsandbox
        POST_BUILD
        COMMAND mkdir -p ${CMAKE_BINARY_DIR}/cmake
    )
    add_custom_command(TARGET mlsandbox
        POST_BUILD
        COMMAND cp ${PROJECT_SOURCE_DIR}/cmake/invoke_setup.py.sh ${CMAKE_BINARY_DIR}/cmake
    )

    add_custom_target(build_python_package
        ALL
        mkdir -p ${CMAKE_BINARY_DIR}/python/MLSandbox
        COMMAND cp -r ${PROJECT_SOURCE_DIR}/python/* ${CMAKE_BINARY_DIR}/python/MLSandbox
        DEPENDS mlsandbox
        COMMENT "Building python package ..."
    )

    # Install all the public header files into the include directory.
    install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/mlsandbox
        DESTINATION ${CMAKE_INSTALL_PREFIX}/include/
        FILES_MATCHING PATTERN "*.hpp"
        PATTERN ".git" EXCLUDE
    )

    # Install the shared libraries into the library directory.
    install(TARGETS mlsandbox
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
    )

    # Build the core python extension module that provides the Python
    # interface to the C++ code of ndhist.
    SET(${PROJECT}_python mlsandbox  ${PYTHON_LIBRARIES} ${BOOST_LIBRARIES} ${BOOSTNUMPY_LIBRARIES})

    #list(APPEND ${PROJECT}_core_LIBRARY_LIST ${BOOSTNUMPY_LIBRARIES})

    add_python_module(MLSandbox ${${PROJECT}_python}
        private/pybindings/module.cxx
        private/pybindings/Distribution.cxx
        private/pybindings/Likelihood.cxx
        private/pybindings/Minimizer.cxx
        private/pybindings/FCRanks.cxx
        private/pybindings/FeldmanCousins.cxx
        private/pybindings/NeymanAnalysis.cxx
    )

    configure_file(
        ${CMAKE_SOURCE_DIR}/cmake/env_shell.sh.in
        ${CMAKE_BINARY_DIR}/env_shell.sh
        @ONLY
    )

    # Install the python module "ndhist" via setuptools.
    configure_file(
        ${CMAKE_SOURCE_DIR}/cmake/install_python_package.cmake.in
        ${CMAKE_BINARY_DIR}/cmake/install_python_package.cmake
        @ONLY
    )
    install(SCRIPT ${CMAKE_BINARY_DIR}/cmake/install_python_package.cmake)

    #enable_testing()
    #add_subdirectory(test)

    message(STATUS "Project \"${PROJECT_NAME}\" configured.")
    message(STATUS "+    Now, type-in \"cd ${CMAKE_BINARY_DIR} && make && make install\"!")
    message(STATUS "+    \"${PROJECT_NAME}\" will be installed in '${CMAKE_INSTALL_PREFIX}'.")
else(${PROJECT_NAME}_CONFIGURED_SUCCESSFULLY)
    message(STATUS "Failed to configure")
endif(${PROJECT_NAME}_CONFIGURED_SUCCESSFULLY)

# ML library source files.

#i3_project(MLSandboxPythonAccess
#      PYTHON_DIR python
#      PYTHON_DEST icecube/MLSandboxPythonAccess
#)

#LIST(APPEND LIB_${PROJECT_NAME}_SOURCEFILES
#                ${LIBRARY_SRC}
#)
#i3_add_library(MLSandboxPythonAccess
#    ${LIB_${PROJECT_NAME}_SOURCEFILES}
#
#    USE_TOOLS boost gsl python
#    USE_PROJECTS  ${LIB_${PROJECT_NAME}_TOOLS}
#)
#
#add_subdirectory(private/pybindings)

#Quick fix to link to thread library
add_definitions(-pthread -O3 -std=c++11)
