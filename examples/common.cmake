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

function (add_examples examples)
    foreach(example ${examples})
        add_executable(${example} "${example}.cpp")
        target_include_directories(${example} PRIVATE ${oneDAL_INCLUDE_DIRS})
        target_link_libraries(${example} PRIVATE ${oneDAL_IMPORTED_TARGETS})
        set_target_properties(${example} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/_cmake_results/intel_intel64_${THREADING_TYPE}_${LINK_TYPE}")
    endforeach()
endfunction()

function (set_link_and_threading_types)
    if ("${TARGET_LINK}" STREQUAL "static")
        set(LINK_TYPE "a" PARENT_SCOPE)
    else()
        set(LINK_TYPE "so" PARENT_SCOPE)
    endif()

    if ("${USE_PARALLEL}" STREQUAL "yes")
        set(THREADING_TYPE "parallel" PARENT_SCOPE)
    else()
        set(THREADING_TYPE "sequential" PARENT_SCOPE)
    endif()
endfunction()

function (change_md_to_mt)
    set(flags
            CMAKE_CXX_FLAGS
            CMAKE_CXX_FLAGS_RELEASE
            CMAKE_C_FLAGS
            CMAKE_C_FLAGS_RELEASE
        PARENT_SCOPE)
    foreach(flag ${flags})
        string(REPLACE "/MD" "/MT" ${flag} "${${flag}}")
    endforeach()
endfunction()
