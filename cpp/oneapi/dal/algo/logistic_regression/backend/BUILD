package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_test_suite",
)

dal_module(
    name = "backend",
    dal_deps = [
        "@onedal//cpp/oneapi/dal/algo/logistic_regression/backend/cpu",
        "@onedal//cpp/oneapi/dal/algo/logistic_regression/backend/gpu",
    ],
)

dal_module(
    name = "model_impl",
    hdrs = glob(["*.hpp"]),
)
