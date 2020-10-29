package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_test_suite",
)

dal_module(
    name = "linear_kernel",
    auto = True,
    dal_deps = [
        "@onedal//cpp/oneapi/dal:core",
    ],
    extra_deps = [
        "@onedal//cpp/daal/src/algorithms/kernel_function:kernel",
    ]
)

dal_test_suite(
    name = "cpu_tests",
    dpc = False,
    srcs = glob([
        "backend/cpu/*_test.cpp",
        "*_test.cpp",
    ]),
    dal_deps = [
        ":linear_kernel",
    ],
)

dal_test_suite(
    name = "gpu_tests_dpc",
    host = False,
    srcs = glob([
        "backend/gpu/*_test.cpp",
    ]),
    dal_deps = [
        ":linear_kernel",
    ],
    tags = ["gpu", "exclusive"],
)

dal_test_suite(
    name = "tests",
    host_tests = [
        ":cpu_tests",
    ],
    dpc_tests = [
        ":gpu_tests_dpc",
    ],
)
