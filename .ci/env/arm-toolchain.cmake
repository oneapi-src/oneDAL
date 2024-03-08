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


# Set the CMake system name to inform CMake that we are cross-compiling
set(CMAKE_SYSTEM_NAME Linux)

# Set the cross-compilation prefix for the toolchain
set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

# Set the target architecture
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# Set compiler flags and options for ARMv8-A architecture with SVE
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=armv8-a+sve")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a+sve")

# Specify the root directory of the cross-compiler toolchain
set(CMAKE_FIND_ROOT_PATH /usr/bin)

# Specify the search paths for libraries and headers
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)