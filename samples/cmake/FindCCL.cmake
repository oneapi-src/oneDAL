#===============================================================================
# Copyright contributors to the oneDAL project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

find_package(oneCCL REQUIRED)
set(CCL_ROOT $ENV{CCL_ROOT})

find_path(CCL_INCLUDE_DIR
  NAMES "ccl.hpp"
  NO_DEFAULT_PATH
  PATH_SUFFIXES include/oneapi/
  PATHS ${CCL_ROOT})

find_library(CCL_LIBRARY
    NAMES "ccl"
    PATHS ${CCL_ROOT}
    PATH_SUFFIXES lib
    NO_DEFAULT_PATH)

if(NOT CCL_INCLUDE_DIR MATCHES NOTFOUND AND CCL_LIBRARY)
  set(CCL_FOUND TRUE)
endif()

if(NOT DEFINED CCL_FOUND)
  message(
    FATAL_ERROR
    "CCL was not found in ${CCL_ROOT}! Set/check CCL_ROOT environment variable!"
  )
else()
  message(STATUS "Found CCL: " ${CCL_FOUND})
  message(STATUS "CCL_ROOT: .......................... " ${CCL_ROOT})
  message(STATUS "CCL_LIBRARY: ....................... " ${CCL_LIBRARY})
  message(STATUS "CCL_INCLUDE_DIR: ................... " ${CCL_INCLUDE_DIR})
endif()
