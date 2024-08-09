package(default_visibility = ["//visibility:public"])

cc_library(
    name = "headers",
    hdrs = glob([
        "include/*.h",
        "include/oneapi/*.hpp",
        "include/oneapi/mkl/*.hpp",
        "include/oneapi/mkl/blas/*.hpp",
        "include/oneapi/mkl/spblas/*.hpp",
        "include/oneapi/mkl/lapack/*.hpp",
        "include/oneapi/mkl/vm/*.hpp",
        "include/oneapi/mkl/vm/device/*.hpp",
        "include/oneapi/mkl/vm/device/detail/*.hpp",
        "include/oneapi/mkl/rng/*.hpp",
        "include/oneapi/mkl/rng/detail/*.hpp",
        "include/oneapi/mkl/rng/device/*.hpp",
        "include/oneapi/mkl/rng/device/detail/*.hpp"
    ]),
    includes = [
        "include",
        "include/oneapi",
        "include/oneapi/mkl",
        "include/oneapi/mkl/blas",
        "include/oneapi/mkl/spblas",
        "include/oneapi/mkl/lapack",
        "include/oneapi/mkl/vm",
        "include/oneapi/mkl/vm/device",
        "include/oneapi/mkl/vm/device/detail",
        "include/oneapi/mkl/rng",
        "include/oneapi/mkl/rng/device",
        "include/oneapi/mkl/rng/device/detail",
        "include/oneapi/mkl/rng/detail" ],
    defines = [
        "MKL_ILP64"
    ],
)

cc_library(
    name = "mkl_core",
    srcs = [
        "lib/libmkl_core.a",
        "lib/libmkl_intel_ilp64.a",
    ],
    linkopts = [
        "-lpthread",
    ],
    deps = [
        ":headers",
    ]
)

cc_library(
    name = "libmkl_sequential",
    srcs = [
        "lib/libmkl_sequential.a",
    ],
    deps = [
        ":mkl_core",
    ]
)

cc_library(
    name = "mkl_thr",
    srcs = [
        "lib/libmkl_tbb_thread.a",
    ],
    linkopts = [
        "-lpthread",
    ],
    deps = [
        ":headers",
        ":mkl_core",
    ]
)

cc_library(
    name = "mkl_seq",
    deps = [
        ":headers",
        ":mkl_core",
        ":libmkl_sequential",
    ],
)

cc_library(
    name = "headers_dpc",
    hdrs = glob(["include/*.h", "include/*.hpp"]),
    includes = [ "include" ],
)

cc_library(
    name = "mkl_dpc",
    srcs = [
        "lib/libmkl_sycl.a",
    ],
    deps = [
        ":headers_dpc",
    ],
)
