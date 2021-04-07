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
        "test/table_adapter.cpp",
    ],
    dal_deps = [":table"],
    framework = "catch2",
    private = True,
)

dal_test_suite(
    name = "row_accessor_tests",
    srcs = [
        "test/row_accessor.cpp",
    ],
    dal_deps = [":table"],
    framework = "catch2",
)

dal_test_suite(
    name = "interface_tests",
    srcs = [
        "test/column_accessor.cpp",
        "test/common.cpp",
        "test/homogen.cpp",
    ],
    dal_deps = [":table"],
)

dal_test_suite(
    name = "tests",
    tests = [
        ":interface_tests",
        ":private_tests",
        ":row_accessor_tests",
    ],
)
