package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl",
    "dal_test_module",
)

dal_test_module(
    name = "common_base",
    auto = True,
    auto_exclude = [
        "gtest_main.cpp",
        "catch_main.cpp",
        "gbench_main.cpp",
        "common.hpp",
    ],
    dal_deps = [
        "@onedal//cpp/oneapi/dal:common",
        "@onedal//cpp/oneapi/dal:table",
        "@onedal//cpp/oneapi/dal/io:csv",
    ],
    dal_test_deps = [
        "@onedal//cpp/oneapi/dal/test/engine/linalg",
        "@onedal//cpp/oneapi/dal/test/engine/metrics",
    ],
    extra_deps = [
        
        "@fmt//:fmt",
    ],
)

dal_test_module(
    name = "common",
    #auto = True,
    #auto_exclude = [
    #    "gtest_main.cpp",
    #    "catch_main.cpp",
    #    "gbench_main.cpp",
    #],
    hdrs = [
        "common.hpp",
    ],
    dal_test_deps = [
        ":common_base",
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
    name = "gbench_main",
    srcs = [
        "gbench_main.cpp",
    ],
    dal_test_deps = [
        ":common_base",
    ],
    extra_deps = [
       # "@gtest//:gtest",
        "@gbench//:benchmark",
    ],
)

load("@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_test_suite",
)

dal_test_suite(
    name = "func_test",
    framework = "gbench",
    srcs = [
        "func.cpp",
    ],
    dal_deps = [
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
