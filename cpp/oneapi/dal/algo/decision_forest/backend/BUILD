package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_test_suite",
)

dal_module(
    name = "backend",
    dal_deps = [
        "@onedal//cpp/oneapi/dal/algo/decisino_forest/backend/cpu",
        "@onedal//cpp/oneapi/dal/algo/decisino_forest/backend/gpu",
    ],
)
