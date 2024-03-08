package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl",
    "dal_module",
)

dal_module(
    name = "parameters",
    dal_deps = [
        "@onedal//cpp/oneapi/dal/algo/decision_forest/parameters/cpu:cpu",
        "@onedal//cpp/oneapi/dal/algo/decision_forest/parameters/gpu:gpu",
    ],
)
