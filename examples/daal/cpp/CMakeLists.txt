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

project(daal_cpp_examples)

set(CMAKE_BUILD_TYPE Release)

set(USE_DPCPP no)
set(USE_NEW_IFACES no)

function (add_examples examples)
    foreach(example ${examples})
        add_executable(${example} "${example}.cpp")
        target_include_directories(${example} PRIVATE ${oneDAL_INCLUDE_DIRS})
        if (UNIX)
            target_link_libraries(${example} PRIVATE -Wl,--start-group ${oneDAL_IMPORTED_TARGETS} -Wl,--end-group)
        else()
            target_link_libraries(${example} PRIVATE ${oneDAL_IMPORTED_TARGETS})
        endif()
        set_target_properties(${example} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/_cmake_results/intel_intel64_${LINK_TYPE}")
    endforeach()
endfunction()

function (set_link_type)
    if ("${TARGET_LINK}" STREQUAL "static")
        set(LINK_TYPE "a" PARENT_SCOPE)
    else()
        set(LINK_TYPE "so" PARENT_SCOPE)
    endif()
endfunction()

function (change_md_to_mdd)
    set(cxx_flag ${CMAKE_CXX_FLAGS})
    set(cxxr_flag ${CMAKE_CXX_FLAGS_RELEASE})
    set(c_flag ${CMAKE_C_FLAGS})
    set(cr_flag ${CMAKE_C_FLAGS_RELEASE})
    set(flags
            cxx_flag
            cxxr_flag
            c_flag
            cr_flag)
    foreach(flag ${flags})
        string(REPLACE "/MD" "/MDd /debug:none" ${flag} "${${flag}}")
    endforeach()

    set(CMAKE_CXX_FLAGS ${cxx_flag} PARENT_SCOPE)
    set(CMAKE_CXX_FLAGS_RELEASE ${cxxr_flag} PARENT_SCOPE)
    set(CMAKE_C_FLAGS ${c_flag} PARENT_SCOPE)
    set(CMAKE_C_FLAGS_RELEASE ${cr_flag} PARENT_SCOPE)
endfunction()

set_link_type()

find_package(oneDAL REQUIRED)

include_directories(source/utils)

# Define variable to specify the examples or directories to include or exclude
set(EXAMPLES "")

# Split the EXAMPLES option into a list of patterns
string(REPLACE "," ";" EXAMPLES_LIST ${EXAMPLES})

# Initialize the EXAMPLES variable with an empty list
set(EXAMPLES "")

# Initialize the EXCLUDE_PATTERN variable
set(EXCLUDE_PATTERN "source/utils/*")

# Recursively find all the example files in the source directory, using the
# EXCLUDE_PATTERN exclude examples or directories
file(GLOB_RECURSE EXAMPLE_FILES source/*/*.cpp EXCLUDE_PATTERN ${EXCLUDE_PATTERN})

# Convert the file names to executable names
foreach(EXAMPLE_FILE ${EXAMPLE_FILES})
    get_filename_component(EXAMPLE_NAME ${EXAMPLE_FILE} NAME_WE)
    if("${EXAMPLES_LIST}" STREQUAL "")
        # If EXAMPLES_LIST is empty, add all examples to the EXAMPLES variable
        list(APPEND EXAMPLES ${EXAMPLE_NAME})
    else()
        # Otherwise, check if the example is in the EXAMPLES_LIST
        foreach(EXAMPLE_PATTERN ${EXAMPLES_LIST})
            if("${EXAMPLE_NAME}" MATCHES "${EXAMPLE_PATTERN}")
                list(APPEND EXAMPLES ${EXAMPLE_NAME})
            endif()
        endforeach()
    endif()
endforeach()

# Add the examples to the build
add_examples(${EXAMPLES})