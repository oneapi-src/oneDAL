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

dal_module(
    name = "test_utils",
    hdrs = glob(["test/*.hpp"]),
)

dal_test_suite(
    name = "common_tests",
    compile_as = [ "c++" ],
    srcs = glob([
        "common_*_test.cpp",
    ]),
    dal_deps = [
        ":decision_forest",
    ],
    dal_test_deps = [
        ":test_utils",
    ],
)

dal_test_suite(
    name = "cpu_tests",
    compile_as = [ "c++" ],
    srcs = glob([
        "backend/cpu/*_test.cpp",
    ]),
    dal_deps = [
        ":decision_forest",
    ],
    dal_test_deps = [
        ":test_utils",
    ],
)

dal_test_suite(
    name = "gpu_tests_dpc",
    compile_as = [ "dpc++" ],
    srcs = glob([
        "backend/gpu/*_test.cpp",
    ]),
    dal_deps = [
        ":decision_forest",
    ],
    dal_test_deps = [
        ":test_utils",
    ],
    tags = ["gpu"],
)

dal_test_suite(
    name = "tests",
    host_tests = [
        ":common_tests",
        ":cpu_tests",
    ],
    dpc_tests = [
        ":gpu_tests_dpc",
    ],
)
