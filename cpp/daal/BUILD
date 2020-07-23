package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:cc.bzl", "cc_module")

cc_module(
    name = "all_headers",
    # TODO: Remove *.i and *.cl
    hdrs = glob(["**/*.h", "**/*.i", "**/*.cl"]),
    system_includes = [ ".", "include" ],
    defines = [
        "__TBB_NO_IMPLICIT_LINKAGE",
        "DAAL_NOTHROW_EXCEPTIONS",
        "DAAL_HIDE_DEPRECATED",
        "TBB_USE_ASSERT=0",
    ],
    deps = [
        "@micromkl//:headers",
        "@micromkl_dpc//:headers",
    ],
)
