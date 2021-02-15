package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_test_suite",
)

dal_module(
    name = "rbf_kernel",
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
    compile_as = [ "c++" ],
    srcs = glob([
        "backend/cpu/*_test.cpp",
        "*_host_test.cpp",
    ]),
    dal_deps = [
        ":rbf_kernel",
    ],
)

dal_test_suite(
    name = "gpu_tests",
    compile_as = [ "dpc++" ],
    srcs = glob([
        "backend/gpu/*_test.cpp",
        "*_dpc_test.cpp",
    ]),
    dal_deps = [
        ":rbf_kernel",
    ],
    tags = ["gpu", "exclusive"],
)

dal_test_suite(
    name = "tests",
    host_tests = [
        ":cpu_tests",
    ],
    dpc_tests = [
        ":gpu_tests",
    ],
)
