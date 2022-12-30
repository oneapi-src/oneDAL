#===============================================================================
# Copyright 2021 Intel Corporation
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

cmake_minimum_required(VERSION 3.1)

# Add cmake scripts and modules to CMake search path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake")
include(setup_examples)

project(daal_cpp_sycl_examples)

set(CMAKE_BUILD_TYPE Release)

set(USE_DPCPP yes)
set(USE_NEW_IFACES no)

set(CUSTOM_OPTIONS "")

if (WIN32)
    set(TARGET_LINK dynamic)
    set(CMAKE_CXX_COMPILE_OBJECT "<CMAKE_CXX_COMPILER> <DEFINES> <INCLUDES> <FLAGS> /Fo<OBJECT> -c <SOURCE>")
    set(CMAKE_CXX_CREATE_STATIC_LIBRARY "lib <OBJECTS> /out:<TARGET>")
    set(CMAKE_CXX_LINK_EXECUTABLE "<CMAKE_CXX_COMPILER> ${CUSTOM_OPTIONS} <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")
endif()

set_link_type()

find_package(oneDAL REQUIRED)

include_directories(source/utils)

# Initialize the EXCLUDE_LIST variable
set(EXCLUDE_LIST
    "source/utils/*"
)

# Define variable to specify the examples or directories to include or exclude
option(EXAMPLES_LIST "")

generate_examples(EXCLUDE_LIST EXAMPLES_LIST)
