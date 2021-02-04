package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_test_suite",
)

dal_module(
    name = "graph",
    auto = True,
    dal_deps = [
        "@onedal//cpp/oneapi/dal:common",
        "@onedal//cpp/oneapi/dal/util",
    ]
)

_DATA_DEPS = [
    "@onedal//cpp/oneapi/dal/graph:test_data",
]

dal_test_suite(
    name = "graph_tests",
    framework = "catch2",
    srcs = glob([
        "service_functions_test.cpp",
    ]),
    dal_deps = [
        ":graph",
        "@onedal//cpp/oneapi/dal/io",
    ],
    data = _DATA_DEPS,
)

