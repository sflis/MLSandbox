#
# $Id$
#
# Copyright (C)
# 2014 - $Date$
#     Martin Wolf <ndhist@martin-wolf.org>
#
# This file implements cmake configuration for building the NDHist tool.
#
# This file is distributed under the BSD 2-Clause Open Source License
# (See LICENSE file).
#
message(STATUS "Entering 'config.cmake'")

set(BUILD_SHARED_LIBS TRUE)

add_definitions(-fPIC)

link_libraries(stdc++)

set(MLSANDBOX_VERSION_STRING "0.1.0" CACHE STRING "The MLSandbox version." FORCE)

message(STATUS "+    MLSANBOX_VERSION: ${MLSANBOX_VERSION_STRING}")
message(STATUS "+    CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")
