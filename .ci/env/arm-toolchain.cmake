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