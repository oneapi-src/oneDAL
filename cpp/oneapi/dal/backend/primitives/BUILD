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
    ],
    dal_deps = [
        ":common",
    ],
)

dal_test_suite(
    name = "common_tests",
    framework = "catch2",
    srcs = glob([
        "test/*.cpp",
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
    ],
    tests = [
        ":common_tests",
    ]
)
