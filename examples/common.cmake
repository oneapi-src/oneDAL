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
        set(LINK_TYPE "a")
    else()
        set(LINK_TYPE "so")
    endif()

    if ("${USE_PARALLEL}" STREQUAL "yes")
        set(THREADING_TYPE "parallel")
    else()
        set(THREADING_TYPE "sequential")
    endif()
endfunction()

function (change_md_to_mt)
    set(CompilerFlags
        CMAKE_CXX_FLAGS
        CMAKE_CXX_FLAGS_RELEASE
        CMAKE_C_FLAGS
        CMAKE_C_FLAGS_RELEASE)
    foreach(CompilerFlag ${CompilerFlags})
        string(REPLACE "/MD" "/MT" ${CompilerFlag} "${${CompilerFlag}}")
    endforeach()
endfunction()
