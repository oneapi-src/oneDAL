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
