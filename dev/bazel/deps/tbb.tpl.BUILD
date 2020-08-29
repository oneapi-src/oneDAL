package(default_visibility = ["//visibility:public"])

cc_library(
    name = "headers",
    hdrs = glob(["include/**/*.h"]),
    includes = [ "include" ],
)

cc_library(
    name = "tbb_binary",
    srcs = [
        "lib/intel64/gcc4.8/libtbb.so.12",
    ],
    linkopts = [
        "-lpthread",
    ],
)

cc_library(
    name = "tbb",
    deps = [
        ":headers",
        ":tbb_binary",
    ],
)

cc_library(
    name = "tbbmalloc",
    srcs = [
        "lib/intel64/gcc4.8/libtbbmalloc.so.2",
    ],
    deps = [
        ":headers",
    ],
)
