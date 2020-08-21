package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_test_suite",
    "dal_collect_modules",
)

dal_module(
    name = "graph_csv",
    hdrs = glob(["**/*graph*.hpp"]),
    srcs = glob(["**/*graph*.cpp"]),
    dal_deps = [
        "@onedal//cpp/oneapi/dal:core",
    ],
)

IOS = [
    "csv",
]

dal_collect_modules(
    name = "io",
    root = "@onedal//cpp/oneapi/dal/io",
    modules = IOS,
    dal_deps = [
        ":graph_csv",
    ],
)

dal_test_suite(
    name = "tests",
    tests = [],
)
