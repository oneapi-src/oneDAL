package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl",
    "dal_module",
)

dal_module(
    name = "util",
    auto = True,
    dal_deps = [
        "@onedal//cpp/oneapi/dal:common",
    ],
    extra_deps = [
        "@gtest//:gtest",
    ],
    testonly = True,
)
