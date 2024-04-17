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

if(CMAKE_SYSTEM_PROCESSOR STREQUAL CMAKE_HOST_SYSTEM_PROCESSOR)
    # If we are not cross-compiling, we don't currently have any tests to pass
    # here
    return()
endif()

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    # Some of the tests fail when run under emulation. For now, exclude the ones
    # that do segfault. Running on native hardware passes for aarch64, at least,
    # so that is a better test than running through emulation anyway
    set(EXCLUDE_LIST
        ${EXCLUDE_LIST}
        "basic_statistics_dense_online"
    )
endif()
