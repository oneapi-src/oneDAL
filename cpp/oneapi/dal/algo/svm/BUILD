package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl", "dal_module", "dal_public_includes")

dal_public_includes(
    name = "public_includes",
)

dal_module(
    name = "svm",
    dpc = True,
    auto = True,
    deps = [
        "@onedal//cpp/oneapi/dal:core",
        "@onedal//cpp/oneapi/dal:linear_kernel",
        "@onedal//cpp/oneapi/dal:rbf_kernel",
        "@onedal//cpp/daal/src/algorithms/svm:kernel",
    ]
)
