package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_test_suite",
)

dal_module(
    name = "kmeans_init",
    auto = True,
    dal_deps = [
        "@onedal//cpp/oneapi/dal:core",
    ],
    extra_deps = [
        "@onedal//cpp/daal/src/algorithms/kmeans:kernel",
    ]
)

dal_test_suite(
    name = "cpu_tests",
    srcs = [
        "backend/cpu/compute_kernel_dense_test.cpp",
    ],
    dal_deps = [
        ":kmeans_init",
    ],
)

dal_test_suite(
    name = "tests",
    tests = [
        ":cpu_tests",
    ],
)
