package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:daal.bzl", "daal_module")
load("@onedal//dev/bazel:dal.bzl", "dal_test_suite")

daal_module(
    name = "kernel",
    auto = True,
    deps = [
        "@onedal//cpp/daal:core",
        "@onedal//cpp/daal:sycl",
        "@onedal//cpp/daal/src/algorithms/regression:kernel",
        "@onedal//cpp/daal/src/algorithms/dtrees/gbt:kernel",
    ],
)

dal_test_suite(
    name = "unit_tests",
    tags = ["unittest"],
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
