package(default_visibility = ["//visibility:public"])

cc_library(
    name = "headers",
    hdrs = glob(["include/*.h", "include/*.hpp"]),
    includes = [ "include" ],
)

cc_library(
    name = "mkl_dpc",
    srcs = [
        "lib/intel64/libmkl_sycl.a",
    ],
    deps = [
        ":headers",
    ],
)
