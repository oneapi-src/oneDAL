package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_test_suite",
)

dal_module(
    name = "pca",
    auto = True,
    dal_deps = [
        "@onedal//cpp/oneapi/dal:core",
    ],
    extra_deps = [
        "@onedal//cpp/daal/src/algorithms/pca:kernel",
        "@onedal//cpp/daal/src/algorithms/pca/transform:kernel",
    ]
)

dal_test_suite(
    name = "cpu_tests",
    compile_as = [ "c++" ],
    srcs = glob([
        "backend/cpu/test/*.cpp",
    ]),
    dal_deps = [
        ":pca",
    ],
    tags = [
        "backend",
    ]
)

dal_test_suite(
    name = "gpu_tests",
    compile_as = [ "dpc++" ],
    srcs = glob([
        "backend/gpu/test/*.cpp",
    ]),
    dal_deps = [
        ":pca",
    ],
    tags = [
        "backend",
    ]
)

dal_test_suite(
    name = "interface_tests",
    framework = "catch2",
    srcs = glob([
        "test/*.cpp",
    ]),
    dal_deps = [
        ":pca",
    ],
    tags = [
        "interface",
    ]
)

dal_test_suite(
    name = "tests",
    tests = [
        ":cpu_tests",
        ":gpu_tests",
        ":interface_tests",
    ],
)
