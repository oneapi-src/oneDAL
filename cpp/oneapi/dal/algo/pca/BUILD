package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_test_suite",
)

dal_module(
    name = "pca",
    auto = True,
    dal_deps = [
        "@onedal//cpp/oneapi/dal:core",
    ],
    extra_deps = [
        "@onedal//cpp/daal/src/algorithms/pca:kernel",
        "@onedal//cpp/daal/src/algorithms/pca/transform:kernel",
    ]
)

dal_test_suite(
    name = "common_tests",
    dpc = False,
    srcs = glob([
        "common_*_test.cpp",
    ]),
    dal_deps = [
        ":pca",
    ],
)

dal_test_suite(
    name = "cpu_tests",
    dpc = False,
    srcs = glob([
        "backend/cpu/*_test.cpp",
    ]),
    dal_deps = [
        ":pca",
    ],
)

dal_test_suite(
    name = "gpu_tests_dpc",
    host = False,
    srcs = glob([
        "backend/gpu/*_test.cpp",
    ]),
    dal_deps = [
        ":pca",
    ],
    tags = ["gpu", "exclusive"],
)

dal_test_suite(
    name = "tests",
    host_tests = [
        ":common_tests",
        ":cpu_tests",
    ],
    dpc_tests = [
        ":gpu_tests_dpc",
    ],
)
