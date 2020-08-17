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
        "lib/intel64/libonedal_core.a",
    ],
    deps = [
        ":headers",
    ],
)

cc_library(
    name = "thread_static",
    srcs = [
        "lib/intel64/libonedal_thread.a",
    ],
    deps = [
        ":headers",
        "@tbb//:tbb",
        "@tbb//:tbbmalloc",
    ],
)

cc_library(
    name = "sequential_static",
    srcs = [
        "lib/intel64/libonedal_sequential.a",
    ],
    deps = [
        ":headers",
    ],
)

cc_library(
    name = "onedal_static",
    srcs = [
        "lib/intel64/libonedal.a",
    ],
    deps = [
        ":headers",
    ],
)
