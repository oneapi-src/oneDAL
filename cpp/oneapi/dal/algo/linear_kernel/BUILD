package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_test_suite",
)

dal_module(
    name = "linear_kernel",
    auto = True,
    dal_deps = [
        "@onedal//cpp/oneapi/dal:core",
    ],
    extra_deps = [
        "@onedal//cpp/daal/src/algorithms/kernel_function:kernel",
    ]
)

dal_test_suite(
    name = "cpu_tests",
    srcs = [
        "backend/cpu/compute_kernel_test.cpp",
    ],
    dal_deps = [
        ":linear_kernel",
    ],
)

dal_test_suite(
    name = "tests",
    tests = [
        ":cpu_tests",
    ],
)
