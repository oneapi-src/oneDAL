package(default_visibility = ["//visibility:public"])

cc_library(
    name = "headers",
    hdrs = glob(["include/**/*.h"]),
    includes = [ "include" ],
    defines = [
        "MKL_ILP64"
    ],
)

cc_library(
    name = "mkl_core",
    srcs = [
        "lib/libmkl_core.a",
    ],
    linkopts = [
        "-lpthread",
    ],
)

cc_library(
    name = "mkl_intel_ilp64",
    srcs = [
        "lib/libmkl_intel_ilp64.a",
    ],
    deps = [
        ":mkl_core",
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
    name = "mkl_seq",
    deps = [
        ":headers",
        ":mkl_core",
        ":mkl_intel_ilp64",
        ":libmkl_sequential",
    ],
)
