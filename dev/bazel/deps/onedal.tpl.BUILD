package(default_visibility = ["//visibility:public"])

cc_library(
    name = "headers",
    hdrs = glob([
        "include/**/*.h",
        "include/oneapi/**/*.hpp",
    ]),
    includes = [ "include" ],
)

cc_library(
    name = "core_static",
    srcs = [
        "lib/libonedal_core.a",
    ],
    deps = [
        ":headers",
        # TODO: Currently vml_ipp lib depends on TBB, but it shouldn't
        #       Remove TBB from deps once problem with vml_ipp is resolved
        "@tbb//:tbb_binary",
    ],
)

cc_library(
    name = "thread_static",
    srcs = [
        "lib/libonedal_thread.a",
    ],
    deps = [
        ":headers",
        "@tbb//:tbb_binary",
        "@tbb//:tbbmalloc_binary",
        "@mkl//:mkl_dpc",
        "@mkl//:headers",
        "@mkl//:mkl_seq",
        "@mkl//:mkl_thr",
    ],
)

cc_library(
    name = "onedal_sycl",
    srcs = [
        "lib/libonedal_sycl.a",
    ],
    deps = [
        ":headers",
    ],
)

cc_library(
    name = "parameters_static",
    srcs = [
        "lib/libonedal_parameters.a",
    ],
    deps = [
        ":headers",
    ],
)

cc_library(
    name = "onedal_static",
    srcs = [
        "lib/libonedal.a",
    ],
    deps = [
        ":headers",
        ":parameters_static",
    ],
)

cc_library(
    name = "parameters_static_dpc",
    srcs = [
        "lib/libonedal_parameters_dpc.a",
    ],
    deps = [
        ":headers",
    ],
)

cc_library(
    name = "onedal_static_dpc",
    srcs = [
        "lib/libonedal_dpc.a",
    ],
    deps = [
        ":headers",
        ":onedal_sycl",
        ":parameters_static_dpc",
    ],
)

cc_library(
    name = "core_dynamic",
    srcs = [
        "lib/libonedal_core.so",
    ],
    deps = [
        ":headers",
        # TODO: Currently vml_ipp lib depends on TBB, but it shouldn't
        #       Remove TBB from deps once problem with vml_ipp is resolved
        "@tbb//:tbb_binary",
    ],
)

cc_library(
    name = "thread_dynamic",
    srcs = [
        "lib/libonedal_thread.so",
    ],
    deps = [
        ":headers",
        "@tbb//:tbb_binary",
        "@tbb//:tbbmalloc_binary",
        "@mkl//:mkl_dpc",
        "@mkl//:headers",
        "@mkl//:mkl_seq",
        "@mkl//:mkl_thr",
    ],
)

cc_library(
    name = "parameters_dynamic",
    srcs = [
        "lib/libonedal_parameters.so",
    ],
    deps = [
        ":headers",
    ],
)

cc_library(
    name = "onedal_dynamic",
    srcs = [
        "lib/libonedal.so",
    ],
    deps = [
        ":headers",
        ":parameters_dynamic",
    ],
)

cc_library(
    name = "parameters_dynamic_dpc",
    srcs = [
        "lib/libonedal_parameters_dpc.so",
    ],
    deps = [
        ":headers",
    ],
)

cc_library(
    name = "onedal_dynamic_dpc",
    srcs = [
        "lib/libonedal_dpc.so",
    ],
    deps = [
        ":headers",
        ":onedal_sycl",
        ":parameters_dynamic_dpc",
    ],
)
