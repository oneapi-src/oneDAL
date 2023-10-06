package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_test_suite",
)

dal_module(
    name = "kernel",
    auto = True,
    dal_deps = [
        "@onedal//cpp/daal:core",
        "@onedal//cpp/daal:sycl",
        "@onedal//cpp/daal/src/algorithms/regression:kernel",
        "@onedal//cpp/daal/src/algorithms/dtrees/gbt:kernel",
    ],
)

dal_test_suite(
    name = "unit_tests",
    framework = "catch2",
    compile_as = [ "c++" ],
    private = True,
    srcs = glob([
        "test/*unit.cpp",
    ]),
    dal_deps = [
        "@onedal//cpp/daal/src/algorithms/regression:kernel",
        "@onedal//cpp/daal/src/algorithms/dtrees/gbt:kernel",
    ]
)