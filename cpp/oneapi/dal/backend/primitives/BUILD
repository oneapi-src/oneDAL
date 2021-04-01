package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_test_suite",
    "dal_collect_modules",
    "dal_collect_test_suites",
)

dal_module(
    name = "common",
    auto = True,
    dal_deps = [
        "@onedal//cpp/oneapi/dal:common",
        "@onedal//cpp/oneapi/dal:table",
    ],
)

dal_collect_modules(
    name = "primitives",
    root = "@onedal//cpp/oneapi/dal/backend/primitives",
    modules = [
        "blas",
        "reduction",
        "sort",
    ],
    dal_deps = [
        ":common",
    ],
)

dal_test_suite(
    name = "dpc_compiler_tests",
    compile_as = [ "dpc++" ],
    framework = "catch2",
    srcs = [
        "test/fill_dpc.cpp",
    ],
    dal_deps = [
        ":common",
    ],
)

dal_test_suite(
    name = "common_tests",
    private = True,
    framework = "catch2",
    srcs = glob([
        "test/*.cpp",
    ], exclude=[
        "test/*_dpc.cpp",
    ]),
    dal_deps = [
        ":common",
    ],
)

dal_collect_test_suites(
    name = "tests",
    root = "@onedal//cpp/oneapi/dal/backend/primitives",
    modules = [
        "blas",
        "reduction",
        "stat",
        "sort",
    ],
    tests = [
        ":common_tests",
        ":dpc_compiler_tests",
    ]
)
