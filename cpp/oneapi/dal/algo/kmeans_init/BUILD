package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_public_includes",
    "dal_test_suite",
)

dal_public_includes(
    name = "public_includes",
)

dal_module(
    name = "kmeans_init",
    dpc = True,
    auto = True,
    deps = [
        "@onedal//cpp/oneapi/dal:core",
        "@onedal//cpp/daal/src/algorithms/kmeans:kernel",
    ]
)

dal_test_suite(
    name = "cpu_tests",
    srcs = [
        "backend/cpu/compute_kernel_dense_test.cpp",
    ],
    deps = [
        ":kmeans_init",
    ],
)
