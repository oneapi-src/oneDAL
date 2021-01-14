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
    name = "cpu_tests",
    srcs = glob([
        "backend/cpu/test/*.cpp",
    ]),
    dal_deps = [
        ":pca",
    ],
    tags = [
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
        ":pca",
    ],
    tags = [
        "gpu-only",
    ]
)

dal_test_suite(
    name = "tests",
    framework = "catch2",
    srcs = glob([
        "test/*.cpp",
    ]),
    dal_deps = [
        ":pca",
        "@onedal//cpp/oneapi/dal/test/engine:datasets",
        "@onedal//cpp/oneapi/dal/test/util",
        "@onedal//cpp/oneapi/dal/backend/linalg",
    ],
    tests = [
        ":cpu_tests",
        ":gpu_tests_dpc",
    ],
)
