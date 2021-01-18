package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl",
    "dal_module",
)

dal_module(
    name = "common",
    hdrs = [
        "common.hpp",
        "macro.hpp",
        "config.hpp",
    ],
    srcs = [
        "config.cpp",
    ],
    dpc_srcs = [
        "common_dpc.cpp",
    ],
    dal_deps = [
        "@onedal//cpp/oneapi/dal:common",
    ],
    extra_deps = [
        "@catch2//:catch2",
        "@fmt//:fmt",
    ],
    testonly = True,
)

dal_module(
    name = "dataframes",
    hdrs = glob(["dataframes.hpp"]),
    srcs = glob(["dataframes.cpp"]),
    dal_deps = [
        ":common",
        "@onedal//cpp/oneapi/dal:table",
    ],
    testonly = True,
)

dal_module(
    name = "gtest_main",
    srcs = [
        "gtest_main.cpp",
    ],
    dal_deps = [
        ":common",
    ],
    extra_deps = [
        "@gtest//:gtest",
    ],
    testonly = True,
)

dal_module(
    name = "catch2_main",
    srcs = [
        "catch_main.cpp",
    ],
    dal_deps = [
        ":common",
    ],
    extra_deps = [
        "@catch2//:catch2",
    ],
    testonly = True,
)
