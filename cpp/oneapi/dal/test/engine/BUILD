package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl",
    "dal_test_module",
)

dal_test_module(
    name = "common",
    auto = True,
    auto_exclude = [
        "gtest_main.cpp",
        "catch_main.cpp",
    ],
    dal_deps = [
        "@onedal//cpp/oneapi/dal:common",
        "@onedal//cpp/oneapi/dal:table",
    ],
    dal_test_deps = [
        "@onedal//cpp/oneapi/dal/test/engine/linalg",
    ],
    extra_deps = [
        "@catch2//:catch2",
        "@fmt//:fmt",
    ],
)

dal_test_module(
    name = "gtest_main",
    srcs = [
        "gtest_main.cpp",
    ],
    dal_test_deps = [
        ":common",
    ],
    extra_deps = [
        "@gtest//:gtest",
    ],
)

dal_test_module(
    name = "catch2_main",
    srcs = [
        "catch_main.cpp",
    ],
    dal_test_deps = [
        ":common",
    ],
    extra_deps = [
        "@catch2//:catch2",
    ],
)
