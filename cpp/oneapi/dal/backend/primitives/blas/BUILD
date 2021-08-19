package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_test_suite",
)

dal_module(
    name = "blas",
    auto = True,
    dal_deps = [
        "@onedal//cpp/oneapi/dal/backend/primitives:common",
    ],
)

dal_test_suite(
    name = "perf_gbench",
    framework = "gbench",
    compile_as = [ "dpc++" ],
    private = True,
    srcs = glob([
        "test/*gbench*.cpp",
    ]),
    dal_deps = [
        ":blas",
        "@onedal//cpp/oneapi/dal/test/engine:common",
    ],
)

dal_test_suite(
    name = "tests",
    private = True,
    framework = "catch2",
    compile_as = [ "dpc++" ],
    srcs = glob([
        "test/*_dpc.cpp",
    ], exclude=[
        "test/*perf*.cpp",
        "test/*gbench*.cpp",
    ]),
    dal_deps = [
        ":blas",
    ],
)