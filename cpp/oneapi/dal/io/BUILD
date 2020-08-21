package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_test_suite",
    "dal_collect_modules",
)

IOS = [
    "csv",
]

dal_collect_modules(
    name = "ios",
    root = "@onedal//cpp/oneapi/dal/io",
    modules = IOS,
)

dal_module(
    name = "io",
    auto = True,
    dal_deps = [
        ":ios",
        "@onedal//cpp/oneapi/dal:common",
        "@onedal//cpp/oneapi/dal/table",
        "@onedal//cpp/oneapi/dal/graph",
        "@onedal//cpp/oneapi/dal/util",
    ],
)

dal_test_suite(
    name = "tests",
    tests = [],
)
