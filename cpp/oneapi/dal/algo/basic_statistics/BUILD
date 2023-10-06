package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_test_suite",
)

dal_module(
    name = "basic_statistics",
    auto = True,
    dal_deps = [
        "@onedal//cpp/oneapi/dal:core",
        "@onedal//cpp/oneapi/dal/backend/primitives:common",
        "@onedal//cpp/oneapi/dal/backend/primitives:reduction",
    ],
    extra_deps = [
        "@onedal//cpp/daal/src/algorithms/low_order_moments:kernel",
    ]
)

dal_test_suite(
    name = "interface_tests",
    framework = "catch2",
    hdrs = glob([
        "test/*.hpp",
    ]),

    srcs = glob([
        "test/*.cpp",
    ]),

    dal_deps = [
        ":basic_statistics",
    ],
)

dal_test_suite(
    name = "tests",
    tests = [
        ":interface_tests",
    ],
)
