package(default_visibility = ["//visibility:public"])

cc_library(
    name = "headers",
    hdrs = glob([
        "include/*.h",
        "include/oneapi/*.hpp",
        "include/oneapi/mkl/*.hpp",
        "include/oneapi/mkl/blas/*.hpp",
        "include/oneapi/mkl/vm/*.hpp",
        "include/oneapi/mkl/rng/*.hpp",
        "include/oneapi/mkl/rng/detail/*.hpp"
    ]),
    includes = [
        "include",
        "include/oneapi",
        "include/oneapi/mkl",
        "include/oneapi/mkl/blas",
        "include/oneapi/mkl/vm",
        "include/oneapi/mkl/rng",
        "include/oneapi/mkl/rng/detail" ],
)

cc_library(
    name = "vml_ipp",
    srcs = [
        "lib/intel64/libmkl_tbb_thread.a",
    ],
    deps = [
        ":headers",
    ],
)

cc_library(
    name = "mkl_thr",
    srcs = [
        "lib/intel64/libmkl_tbb_thread.a",
    ],
    deps = [
        ":headers",
    ],
)
