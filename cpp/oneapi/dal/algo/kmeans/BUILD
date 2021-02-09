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
        "backend/cpu/test/*.cpp",
    ]),
    dal_deps = [
        ":kmeans",
    ],
    tags = [
        "backend",
        "cpu-only",
    ]
)

dal_test_suite(
    name = "gpu_tests_dpc",
    host = False,
    srcs = glob([
        "backend/gpu/test/*.cpp",
    ]),
    dal_deps = [
        ":kmeans",
    ],
    tags = [
        "backend",
        "gpu-only",
    ]
)

dal_test_suite(
    name = "interface_tests",
    framework = "catch2",
    srcs = glob([
        "test/*.cpp",
    ]),
    dal_deps = [
        ":kmeans",
    ],
    tags = [
        "interface",
    ]
)

dal_test_suite(
    name = "tests",
    tests = [
        ":cpu_tests",
        ":gpu_tests_dpc",
        ":interface_tests",
    ],
)

