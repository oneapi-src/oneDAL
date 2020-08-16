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
    name = "rbf_kernel",
    dpc = True,
    auto = True,
    deps = [
        "@onedal//cpp/oneapi/dal:core",
        "@onedal//cpp/daal/src/algorithms/kernel_function:kernel",
    ]
)

dal_test_suite(
    name = "cpu_tests",
    srcs = [
        "backend/cpu/compute_kernel_test.cpp",
    ],
    deps = [
        ":rbf_kernel",
    ],
)
