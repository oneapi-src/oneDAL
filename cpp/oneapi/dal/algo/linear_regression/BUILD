package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_test_suite",
)

dal_module(
    name = "linear_regression",
    auto = True,
    dal_deps = [
        "@onedal//cpp/oneapi/dal:core",
        "@onedal//cpp/oneapi/dal/backend/primitives:common",
    ],
    extra_deps = [
        "@onedal//cpp/daal:core",
        "@onedal//cpp/daal/src/algorithms/linear_model:kernel",
        "@onedal//cpp/daal/src/algorithms/linear_regression:kernel",
    ],
)

dal_test_suite(
    name = "interface_tests",
    framework = "catch2",
    srcs = glob([
        "test/*.cpp",
    ]),
    dal_deps = [
        ":linear_regression",
    ],
)

dal_test_suite(
    name = "tests",
    tests = [
        ":interface_tests",
    ],
)
