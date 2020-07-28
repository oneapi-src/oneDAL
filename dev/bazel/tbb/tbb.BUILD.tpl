package(default_visibility = ["//visibility:public"])
# load("@onedal//dev/bazel:cc.bzl", "cc_module")

cc_library(
    name = "headers",
    hdrs = glob(["include/**/*.h"]),
    includes = [ "include" ],
)

cc_library(
    name = "tbb",
    srcs = [
        "lib/intel64/gcc4.8/libtbb.so",
    ],
    deps = [
        ":headers",
    ],
    linkopts = [
        "-lpthread",
    ],
)

cc_library(
    name = "tbbmalloc",
    srcs = [
        "lib/intel64/gcc4.8/libtbbmalloc.so",
    ],
    deps = [
        ":headers",
    ],
)
