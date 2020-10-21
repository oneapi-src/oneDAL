package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_test_suite",
    "dal_global_header_test",
)

dal_module(
    name = "dal_header",
    hdrs = [
        "dal.hpp",
    ],
)

dal_global_header_test(
    name = "dal_hpp_test",
    algo_dir = "dal/algo",
    algo_exclude = [
        "jaccard",
    ],
    dal_deps = [
        ":dal_header",
        "@onedal//cpp/oneapi/dal:core",
        "@onedal//cpp/oneapi/dal:optional",
    ],
)

dal_test_suite(
    name = "tests",
    tests = [
        "dal_hpp_test",
    ],
)
