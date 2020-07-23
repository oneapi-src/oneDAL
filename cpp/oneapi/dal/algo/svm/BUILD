package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl", "dal_module")

dal_module(
    name = "svm",
    dpc = True,
    auto = True,
    deps = [
        "@onedal//cpp/oneapi/dal:core",
        "@onedal//cpp/oneapi/dal/algo:linear_kernel",
        "@onedal//cpp/oneapi/dal/algo:rbf_kernel",
    ]
)
