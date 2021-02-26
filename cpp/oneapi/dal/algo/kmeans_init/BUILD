package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_test_suite",
)

dal_module(
    name = "kmeans_init",
    auto = True,
    dal_deps = [
        "@onedal//cpp/oneapi/dal:core",
    ],
    extra_deps = [
        "@onedal//cpp/daal/src/algorithms/kmeans:kernel",
    ]
)

dal_test_suite(
    name = "common_tests",
    compile_as = [ "c++" ],
    srcs = glob([
        "common_*_test.cpp",
    ]),
    dal_deps = [
        ":kmeans_init",
    ],
)

dal_test_suite(
    name = "cpu_tests",
    compile_as = [ "c++" ],
    srcs = glob([
        "backend/cpu/*_test.cpp",
    ]),
    dal_deps = [
        ":kmeans_init",
    ],
)

dal_test_suite(
    name = "tests",
    host_tests = [
        ":common_tests",
        ":cpu_tests",
    ],
)
