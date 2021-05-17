load(
    "@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_test_suite",
)

package(default_visibility = ["//visibility:public"])

dal_module(
    name = "distributed",
    auto = True,
    dal_deps = [
        "@onedal//cpp/oneapi/dal:common",
    ],
    extra_deps = [
        "@mpi//:mpi",
    ],
)

dal_test_suite(
    name = "tests",
    srcs = glob([
        "test/*.cpp",
    ]),
    dal_deps = [":distributed"],
    framework = "catch2",
)
