package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_test_suite",
)

dal_module(
    name = "table",
    auto = True,
    dal_deps = [
        "@onedal//cpp/oneapi/dal:common",
        "@onedal//cpp/oneapi/dal/util",
    ]
)

dal_test_suite(
    name = "table_tests",
    srcs = [
        "common_test.cpp",
        "homogen_test.cpp",
    ],
    dal_deps = [ ":table" ],
)

dal_test_suite(
    name = "accessor_tests",
    srcs = [
        "row_accessor_test.cpp",
        "column_accessor_test.cpp",
    ],
    dal_deps = [ ":table" ],
)

dal_test_suite(
    name = "builder_tests",
    srcs = [
        "detail/table_builder_test.cpp",
    ],
    dal_deps = [ ":table" ],
)

dal_test_suite(
    name = "tests",
    tests = [
        ":table_tests",
        ":accessor_tests",
        ":builder_tests",
    ]
)
