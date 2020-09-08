package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_test_suite",
)

dal_module(
    name = "decision_forest",
    auto = True,
    dal_deps = [
        "@onedal//cpp/oneapi/dal:core",
    ],
    extra_deps = [
        "@onedal//cpp/daal/src/algorithms/dtrees/forest/classification:kernel",
        "@onedal//cpp/daal/src/algorithms/dtrees/forest/regression:kernel",
    ]
)

dal_test_suite(
    name = "common_tests",
    srcs = glob([
        "common_*_test.cpp",
    ]),
    dal_deps = [
        ":decision_forest",
    ],
)

dal_test_suite(
    name = "cpu_tests",
    srcs = glob([
        "backend/cpu/*_test.cpp",
    ]),
    dal_deps = [
        ":decision_forest",
    ],
)

dal_test_suite(
    name = "tests",
    tests = [
        ":common_tests",
        ":cpu_tests",
    ],
)
