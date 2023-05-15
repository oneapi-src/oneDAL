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
        "detail/test/homogen_utils.cpp",
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
        "test/common_serialization.cpp",
        "test/homogen_serialization.cpp",
        "detail/test/csr_accessor.cpp",
        "test/csr.cpp",
        "test/column_accessor.cpp",
        "test/common.cpp",
    ],
    dal_deps = [":table"],
    framework = "catch2",
)

dal_test_suite(
    name = "tests",
    tests = [
        ":private_tests",
        ":catch2_tests",
    ],
)
