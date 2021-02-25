cmake_minimum_required(VERSION 3.1)

project(daal_cpp_sycl_examples)

set(CMAKE_BUILD_TYPE Release)

set(USE_DPCPP yes)
set(USE_NEW_IFACES no)

if (WIN32)
    set(TARGET_LINK dynamic)
endif()

include(../../common.cmake)
set_link_and_threading_types()

include(FindPackageHandleStandardArgs)
find_package(oneDAL REQUIRED)

include_directories(source/utils)

add_subdirectory(source/pca)
