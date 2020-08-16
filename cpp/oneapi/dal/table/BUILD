package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_public_includes",
    "dal_test_suite",
)

dal_public_includes(
    name = "public_includes",
)

dal_module(
    name = "table",
    dpc = True,
    auto = True,
    deps = [
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
    deps = [ ":table" ],
)

dal_test_suite(
    name = "accessor_tests",
    srcs = [
        "row_accessor_test.cpp",
        "column_accessor_test.cpp",
    ],
    deps = [ ":table" ],
)

dal_test_suite(
    name = "builder_tests",
    srcs = [
        "detail/table_builder_test.cpp",
    ],
    deps = [ ":table" ],
)
