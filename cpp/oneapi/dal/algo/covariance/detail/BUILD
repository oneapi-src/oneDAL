package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_test_suite",
)

dal_module(
    name = "detail",
    auto=True,
    dal_deps = [
        "@onedal//cpp/oneapi/dal/algo/covariance:core",
        "@onedal//cpp/oneapi/dal/algo/covariance/backend/cpu",
        "@onedal//cpp/oneapi/dal/algo/covariance/backend/gpu",
        "@onedal//cpp/oneapi/dal/algo/covariance/parameters",
    ]
)
