package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_test_suite",
)

dal_module(
    name = "svm",
    auto = True,
    dal_deps = [
        "@onedal//cpp/oneapi/dal:core",
        "@onedal//cpp/oneapi/dal/algo:linear_kernel",
        "@onedal//cpp/oneapi/dal/algo:rbf_kernel",
    ],
    extra_deps = [
        "@onedal//cpp/daal/src/algorithms/svm:kernel",
    ]
)

dal_test_suite(
    name = "cpu_tests",
    srcs = glob([
        "backend/cpu/*_test.cpp",
    ]),
    dal_deps = [
        ":svm",
    ],
)

dal_test_suite(
    name = "gpu_tests",
    host = False,
    srcs = glob([
        "backend/gpu/*_test.cpp",
    ]),
    dal_deps = [
        ":svm",
    ],
    tags = ["gpu", "exclusive"],
)

dal_test_suite(
    name = "tests",
    tests = [
        ":cpu_tests",
        ":gpu_tests",
    ],
)
