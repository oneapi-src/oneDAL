package(default_visibility = ["//visibility:public"])

cc_library(
    name = "headers",
    hdrs = glob(["include/*.h", "%{os}/include/*.h"]),
    includes = [ "include", "%{os}/include" ],
)

cc_library(
    name = "vml_ipp",
    srcs = [
        "%{os}/lib/libdaal_vmlipp_core.a",
    ],
    deps = [
        ":headers",
    ],
)

cc_library(
    name = "mkl_thr",
    srcs = [
        "%{os}/lib/libdaal_mkl_thread.a",
    ],
    deps = [
        ":headers",
    ],
)
