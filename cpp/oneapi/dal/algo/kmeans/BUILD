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
    dpc = False,
    srcs = glob([
        "backend/cpu/*_test.cpp",
    ]),
    dal_deps = [
        ":kmeans",
    ],
)

dal_test_suite(
    name = "gpu_tests_dpc",
    host = False,
    srcs = glob([
        "backend/gpu/*_test.cpp",
    ]),
    dal_deps = [
        ":kmeans",
    ],
    tags = ["gpu", "exclusive"],
)

dal_test_suite(
    name = "tests",
    host_tests = [
        ":cpu_tests",
    ],
    dpc_tests = [
        ":gpu_tests_dpc",
    ],
)
