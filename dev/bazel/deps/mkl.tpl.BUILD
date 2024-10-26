package(default_visibility = ["//visibility:public"])

cc_library(
    name = "headers",
    hdrs = glob([
        "include/**/*.h",
        "include/**/*.hpp",
    ]),
    includes = [
        "include",
    ],
    defines = [
        "MKL_ILP64"
    ],
)

cc_library(
    name = "mkl_core",
    srcs = [
        "lib/libmkl_core.a",
        "lib/libmkl_intel_ilp64.a",
        "lib/libmkl_tbb_thread.a",
    ],
    linkopts = [
        "-lpthread",
    ],
    deps = [
        ":headers",
    ]
)

cc_library(
    name = "mkl_thr",
    linkopts = [
        "-lpthread",
    ],
    deps = [
        ":headers",
        ":mkl_core",
    ]
)

cc_library(
    name = "mkl_dpc",
    # TODO: add a mechanism to get attr from bazel command(it's not available for now)
    linkopts = [
        # Currently its hardcoded to 16 to get the best trade-off between linking speedup and resources used.
        # If the number of processors on machine is below 16 it will be defaulted to `nproc`.
        "-fsycl-max-parallel-link-jobs=16",
    ],
    srcs = [
        "lib/libmkl_sycl.a",
    ],
    deps = [
        ":headers",
    ],
)
