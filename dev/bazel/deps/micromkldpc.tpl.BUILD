package(default_visibility = ["//visibility:public"])

cc_library(
    name = "headers",
    hdrs = glob(["include/*.h", "include/*.hpp"]),
    includes = [ "include" ],
)

cc_library(
    name = "mkl_dpc",
    srcs = [
        "lib/libdaal_sycl.a",
    ],
    deps = [
        ":headers",
        "@opencl//:opencl_binary",
    ],
)
