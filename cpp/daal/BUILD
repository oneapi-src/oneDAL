package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:cc.bzl", "cc_module")

cc_module(
    name = "all_headers",
    hdrs = glob(["**/*.h", "**/*.i"]),
    system_includes = [ ".", "include" ],
    deps = [
        "@micromkl//:headers",
    ],
)
