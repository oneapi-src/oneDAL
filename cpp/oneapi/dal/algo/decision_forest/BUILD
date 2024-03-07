package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_test_suite",
)

dal_module(
    name = "core",
    auto = True,
    dal_deps = [
        "@onedal//cpp/oneapi/dal:core",
    ],
)

dal_module(
    name = "parameters",
    dal_deps = [
        "@onedal//cpp/oneapi/dal/algo/covariance/parameters",
    ],
)

dal_module(
    name = "decision_forest",
    dal_deps = [
        ":core",
        ":parameters",
        "@onedal//cpp/oneapi/dal/algo/decision_forest/detail",
        "@onedal//cpp/oneapi/dal/algo/decision_forest/backend",
        "@onedal//cpp/oneapi/dal/backend/primitives:common",
        "@onedal//cpp/oneapi/dal/backend/primitives:sort",
        "@onedal//cpp/oneapi/dal/backend/primitives:rng",
    ],
    extra_deps = [
        "@onedal//cpp/daal/src/algorithms/dtrees/forest/classification:kernel",
        "@onedal//cpp/daal/src/algorithms/dtrees/forest/regression:kernel",
    ]
)

dal_test_suite(
    name = "interface_tests",
    framework = "catch2",
    hdrs = glob([
        "test/*.hpp",
    ]),
    srcs = glob([
        "test/*.cpp",
    ]),
    dal_deps = [
        ":core",
        ":parameters",
        ":decision_forest",
    ],
)

dal_test_suite(
    name = "tests",
    tests = [
        ":core",
        ":parameters",
        ":decision_forest",
        ":interface_tests",
    ],
)
