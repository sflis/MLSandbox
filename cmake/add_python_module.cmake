#
# $Id$
#
# Copyright (C)
# 2014 - $Date$
#     Martin Wolf <ndhist@martin-wolf.org>
#
# This file implements the cmake
#     add_python_module(<name> <additional_library_list> src1 src2 ... srcN)
# function, which is used to build modules for python.
#
# This file is distributed under the BSD 2-Clause Open Source License
# (See LICENSE file).
#
function(add_python_module _NAME _ADD_LIB_LIST)

    option(PYTHON_ENABLE_MODULE_${_NAME}
        "Add module ${_NAME}" TRUE)

    get_property(_TARGET_SUPPORTS_SHARED_LIBS
        GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS)
    option(PYTHON_MODULE_${_NAME}_BUILD_SHARED
        "Add module ${_NAME} shared" ${_TARGET_SUPPORTS_SHARED_LIBS})

    # Mark these options as advanced.
    mark_as_advanced(
        PYTHON_ENABLE_MODULE_${_NAME}
        PYTHON_MODULE_${_NAME}_BUILD_SHARED)

    if(PYTHON_ENABLE_MODULE_${_NAME})

        if(PYTHON_MODULE_${_NAME}_BUILD_SHARED)
            set(PY_MODULE_TYPE MODULE)
            set_property(GLOBAL APPEND PROPERTY PY_MODULES_LIST ${_NAME})
        else()
            set(PY_MODULE_TYPE STATIC)
            set_property(GLOBAL APPEND PROPERTY PY_STATIC_MODULES_LIST ${_NAME})
        endif()

        add_library(${_NAME} ${PY_MODULE_TYPE} ${ARGN})
        target_link_libraries(${_NAME} ${PYTHON_LIBRARIES} ${_ADD_LIB_LIST})

        get_target_property(${_NAME}_LIBOUTDIR
            ${_NAME} LIBRARY_OUTPUT_DIRECTORY)

        if(PYTHON_MODULE_${_NAME}_BUILD_SHARED)
            set_target_properties(${_NAME} PROPERTIES PREFIX "")
            set_target_properties(${_NAME} PROPERTIES SUFFIX ${PYTHON_MODULE_EXTENSION})
        endif()

        # Copy the built python module into the ndhist python package directory
        # within the build tree, so it can be installed by setuptools during
        # the install stage.
        add_dependencies(build_python_package ${_NAME})
        add_custom_command(TARGET build_python_package
            POST_BUILD
            COMMAND cp ${${_NAME}_LIBOUTDIR}/${_NAME}${PYTHON_MODULE_EXTENSION} ${CMAKE_BINARY_DIR}/python/MLSandbox.so
            COMMENT "Copying python extension module \"${_NAME}\""
        )

    endif()
endfunction(add_python_module)
