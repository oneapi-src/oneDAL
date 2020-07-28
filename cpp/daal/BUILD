package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:daal.bzl", "daal_module")

daal_module(
    name = "includes",
    # TODO: Remove *.i and *.cl
    hdrs = glob(["include/**/*.h"]),
    system_includes = [ ".", "include" ],
    defines = [
        "DAAL_NOTHROW_EXCEPTIONS",
        "DAAL_HIDE_DEPRECATED",
    ],
    deps = [
        "@micromkl//:headers",
        "@micromkl_dpc//:headers",
    ],
)
