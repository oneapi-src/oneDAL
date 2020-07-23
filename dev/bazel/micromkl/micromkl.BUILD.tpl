package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:cc.bzl", "cc_module")

cc_module(
    name = "headers",
    hdrs = glob(["include/*.h"]),
    system_includes = [ "include" ],
)

cc_module(
    name = "mkl_seq",
    libs = [
        "libdaal_mkl_sequential.a",
    ],
    deps = [
        ":headers",
    ],
)

cc_module(
    name = "mkl_thr",
    libs = [
        "libdaal_mkl_thread.a",
    ],
    deps = [
        ":headers",
    ],
)

cc_module(
    name = "vml_ipp",
    libs = [
        "libdaal_vmlipp_core.a",
    ],
    deps = [
        ":headers",
    ],
)
