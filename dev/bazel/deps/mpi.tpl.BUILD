package(default_visibility = ["//visibility:public"])

cc_library(
    name = "headers",
    hdrs = glob(["include/**/*.h"]),
    includes = [ "include" ],
)

cc_library(
    name = "libmpi",
    srcs = [
        "lib/release/libmpi.so.12",
    ],
)

cc_library(
    name = "libfabric",
    srcs = [
        "libfabric/lib/libfabric.so.1",
    ],
)

cc_library(
    name = "mpi",
    deps = [
        ":headers",
        ":libmpi",
        ":libfabric",
    ],
)
