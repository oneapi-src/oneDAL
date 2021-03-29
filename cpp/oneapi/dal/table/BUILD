load(
    "@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_test_suite",
)

package(default_visibility = ["//visibility:public"])

dal_module(
    name = "table",
    auto = True,
    dal_deps = [
        "@onedal//cpp/oneapi/dal:common",
        "@onedal//cpp/oneapi/dal/util",
    ],
)

dal_test_suite(
    name = "private_tests",
    srcs = [
        # TODO: Move conversion tests to backend folder
        "test/convert.cpp",
        "test/table_builder.cpp",
    ],
    dal_deps = [":table"],
    framework = "catch2",
    private = True,
)

dal_test_suite(
    name = "catch2_tests",
    srcs = [
        "test/row_accessor.cpp",
        "test/homogen.cpp",
    ],
    dal_deps = [":table"],
    framework = "catch2",
)

dal_test_suite(
    name = "gtest_tests",
    srcs = [
        "test/column_accessor.cpp",
        "test/common.cpp",
    ],
    dal_deps = [":table"],
)

dal_test_suite(
    name = "tests",
    tests = [
        ":private_tests",
        ":gtest_tests",
        ":catch2_tests",
    ],
)
