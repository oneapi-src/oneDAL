package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_test_suite",
)

dal_module(
    name = "kmeans",
    auto = True,
    dal_deps = [
        "@onedal//cpp/oneapi/dal:core",
    ],
    extra_deps = [
        "@onedal//cpp/daal/src/algorithms/kmeans:kernel",
    ],
)

dal_test_suite(
    name = "cpu_tests",
    srcs = [
        "backend/cpu/train_kernel_lloyd_dense_test.cpp",
    ],
    dal_deps = [
        ":kmeans",
    ],
)

dal_test_suite(
    name = "tests",
    tests = [
        ":cpu_tests",
    ],
)
