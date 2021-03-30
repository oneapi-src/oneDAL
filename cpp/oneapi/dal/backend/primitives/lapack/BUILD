package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_test_suite",
)

dal_module(
    name = "lapack",
    auto = True,
    dal_deps = [
        "@onedal//cpp/oneapi/dal:common", # Needed for LAPACK from old interfaces
        "@onedal//cpp/oneapi/dal/backend/primitives:common",
    ],
)

dal_test_suite(
    name = "tests",
    framework = "catch2",
    private = True,
    srcs = glob([
        "test/*.cpp",
    ]),
    dal_deps = [
        ":lapack",
    ],
)
