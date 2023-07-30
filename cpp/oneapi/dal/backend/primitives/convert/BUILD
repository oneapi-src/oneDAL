package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_test_suite",
)

dal_module(
    name = "convert",
    auto = True,
    dal_deps = [
        "@onedal//cpp/oneapi/dal:core",
        "@onedal//cpp/oneapi/dal/backend/primitives:common",
    ],
)

dal_test_suite(
    name = "perf_tests",
    framework = "catch2",
    private = True,
    srcs = glob([
        "test/*perf*.cpp",
    ]),
    dal_deps = [
        ":convert",
    ],
)

dal_test_suite(
    name = "tests",
    framework = "catch2",
    private = True,
    srcs = glob([
        "test/*.cpp",
    ], exclude=[
        "test/*perf*.cpp",
    ]),
    dal_deps = [
        ":convert",
    ],
)