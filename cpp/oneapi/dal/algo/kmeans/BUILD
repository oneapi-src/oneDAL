package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_test_suite",
)

dal_module(
    name = "kmeans",
    auto = True,
    dal_deps = [
        "@onedal//cpp/oneapi/dal:core",
        "@onedal//cpp/oneapi/dal/backend/primitives:selection",
    ],
    extra_deps = [
        "@onedal//cpp/daal/src/algorithms/kmeans:kernel",
    ],
)

dal_test_suite(
    name = "cpu_tests",
    private = True,
    compile_as = [ "c++" ],
    srcs = glob([
        "backend/cpu/test/*.cpp",
    ]),
    dal_deps = [
        ":kmeans",
    ],
)

dal_test_suite(
    name = "gpu_tests",
    private = True,
    compile_as = [ "dpc++" ],
    srcs = glob([
        "backend/gpu/test/*.cpp",
    ]),
    dal_deps = [
        ":kmeans",
    ],
)

dal_test_suite(
    name = "interface_tests",
    framework = "catch2",
    srcs = glob([
        "test/*.cpp",
    ]),
    dal_deps = [
        ":kmeans",
    ],
)

dal_test_suite(
    name = "unit_tests",
    framework = "catch2",
    private = True,
    compile_as = [ "dpc++" ],
    srcs = glob([
        "test/unit/*.cpp",
    ]),
    dal_deps = [
        ":kmeans",
    ],
)

dal_test_suite(
    name = "tests",
    tests = [
        ":cpu_tests",
        ":gpu_tests",
        ":interface_tests",
    ],
)
