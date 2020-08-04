package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:daal.bzl",
    "daal_module",
    "daal_static_lib",
    "daal_generate_version",
    "daal_generate_kernel_defines",
)

daal_module(
    name = "includes",
    hdrs = glob(["include/**/*.h"]),
    includes = [ ".", "include" ],
    defines = [
        "DAAL_NOTHROW_EXCEPTIONS",
        "DAAL_HIDE_DEPRECATED",
    ],
    deps = [
        "@micromkl//:headers",
        "@micromkl_dpc//:headers",
    ],
)

daal_generate_version(
    name = "version",
    out = "_daal_version_gen.h",
)

daal_generate_kernel_defines(
    name = "kernel_defines",
    out = "_daal_kernel_defines_gen.h",
)

daal_module(
    name = "service_headers",
    hdrs = glob([
        "src/services/**/*.h",
        "src/externals/**/*.h",
        "src/algorithms/service_*.h",
    ]) + [
        ":version",
        ":kernel_defines",
    ],
    deps = [
        "@onedal//cpp/daal:includes",
    ],
)

daal_module(
    name = "threading_headers",
    hdrs = glob(["src/threading/**/*.h"]),
    deps = [
        ":service_headers",
    ],
)

daal_module(
    name = "threading_seq",
    srcs = glob(["src/threading/**/*.cpp"]),
    local_defines = [
        "__DO_SEQ_LAYER__",
    ],
    deps = [
        ":threading_headers",
        "@micromkl//:mkl_seq",
    ],
)

daal_module(
    name = "threading_tbb",
    srcs = glob(["src/threading/**/*.cpp"]),
    local_defines = [
        "__DO_TBB_LAYER__",
        "__TBB_NO_IMPLICIT_LINKAGE",
        "__TBB_LEGACY_MODE",
        "TBB_SUPPRESS_DEPRECATED_MESSAGES",
        "TBB_USE_ASSERT=0",
    ],
    deps = [
        ":threading_headers",
        "@tbb//:tbb",
        "@tbb//:tbbmalloc",
        "@micromkl//:mkl_thr",
    ],
)

daal_module(
    name = "services",
    srcs = glob(
        ["src/services/**/*.cpp", "src/externals/**/*.cpp"],
        exclude = ["src/externals/**/*_win_dll.cpp"],
    ),
    deps = [
        ":service_headers",
        ":threading_headers",
        "@micromkl//:vml_ipp",
    ],
)

daal_module(
    name = "data_management",
    hdrs = glob(["src/data_management/**/*.h"]),
    srcs = glob(["src/data_management/**/*.cpp"]),
    deps = [
        ":services",
    ],
)

daal_module(
    name = "algorithms_base",
    hdrs = glob(["src/algorithms/*.h"], exclude=["src/algorithms/service_*"]),
    srcs = glob(["src/algorithms/*.cpp"], exclude=["src/algorithms/service_*"]),
    deps = [
        ":services",
    ],
)

daal_module(
    name = "sycl",
    hdrs = glob(["src/sycl/**/*.h", "src/sycl/**/*.cl"]),
    srcs = glob(["src/sycl/**/*.cpp"]),
    deps = [
        ":services",
        "@onedal//cpp/daal/src/algorithms/engines",
        "@micromkl_dpc//:headers",
    ],
)

daal_module(
    name = "core",
    deps = [
        ":services",
        ":data_management",
        ":algorithms_base",
    ],
)

daal_module(
    name = "core_sycl",
    deps = [
        ":core",
        ":sycl",
    ],
)

daal_static_lib(
    name = "core_static",
    lib_name = "daal_core",
    deps = [
        ":core",
        ":core_sycl",
        "@onedal//cpp/daal/src/algorithms/kmeans:kmeans",
        "@onedal//cpp/daal/src/algorithms/kmeans:kmeans_init",
    ],
)

daal_static_lib(
    name = "sequential_static",
    lib_name = "daal_sequential",
    deps = [
        ":threading_seq",
    ],
)

daal_static_lib(
    name = "thread_static",
    lib_name = "daal_thread",
    deps = [
        ":threading_tbb",
    ],
)

filegroup(
    name = "all_static",
    srcs = [
        ":core_static",
        ":sequential_static",
        ":thread_static",
    ],
)
