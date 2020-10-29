package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_test_suite",
)

dal_module(
    name = "knn",
    auto = True,
    dal_deps = [
        "@onedal//cpp/oneapi/dal:core",
    ],
    extra_deps = [
        "@onedal//cpp/daal/src/algorithms/k_nearest_neighbors:kernel",
    ],
)

dal_test_suite(
    name = "common_tests",
    dpc = False,
    srcs = glob([
        "common_*_test.cpp",
    ]),
    dal_deps = [
        ":knn",
    ],
)

dal_test_suite(
    name = "gpu_tests_dpc",
    host = False,
    srcs = glob([
        "common_*_test_dpc.cpp",
    ]),
    dal_deps = [
        ":knn",
    ],
    tags = ["gpu"],
)

dal_test_suite(
    name = "tests",
    host_tests = [
        ":common_tests",
    ],
    dpc_tests = [
        ":gpu_tests_dpc",
    ],
)