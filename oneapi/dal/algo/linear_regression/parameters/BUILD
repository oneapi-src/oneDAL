package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_test_suite",
)

dal_module(
    name = "parameters",
    dal_deps = [
        "@onedal//cpp/oneapi/dal/algo/linear_regression/parameters/cpu",
        "@onedal//cpp/oneapi/dal/algo/linear_regression/parameters/gpu",
    ],
)
