/*******************************************************************************
* Copyright 2021 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

cmake_minimum_required(VERSION 3.1)

project(oneapi_dpc_examples)

set(CMAKE_BUILD_TYPE Release)

set(USE_DPCPP yes)
set(USE_NEW_IFACES yes)

if (WIN32)
    set(TARGET_LINK dynamic)
endif()

include(../../common.cmake)
set_link_and_threading_types()

include(FindPackageHandleStandardArgs)
find_package(oneDAL REQUIRED)

include_directories(source)

add_subdirectory(source/pca)
