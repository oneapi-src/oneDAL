cmake_minimum_required(VERSION 3.1)

project(oneapi_cpp_examples)

set(CMAKE_BUILD_TYPE Release)

set(USE_DPCPP no)
set(USE_NEW_IFACES yes)

include(../../common.cmake)
set_link_and_threading_types()
if (WIN32)
    change_md_to_mt()
endif()

include(FindPackageHandleStandardArgs)
find_package(oneDAL REQUIRED)

include_directories(source)

add_subdirectory(source/pca)
