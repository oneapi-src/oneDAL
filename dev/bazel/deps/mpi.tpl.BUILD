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

filegroup(
    name = "fi",
    srcs = [
        "libfabric/lib/prov/libefa-fi.so",
        "libfabric/lib/prov/libmlx-fi.so",
        "libfabric/lib/prov/libpsmx2-fi.so",
        "libfabric/lib/prov/librxm-fi.so",
        "libfabric/lib/prov/libshm-fi.so",
        "libfabric/lib/prov/libsockets-fi.so",
        "libfabric/lib/prov/libtcp-fi.so",
        "libfabric/lib/prov/libverbs-fi.so",
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
