package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_test_suite",
)

dal_module(
    name = "knn",
    auto = True,
    dal_deps = [
        "@onedal//cpp/oneapi/dal:core",
        "@onedal//cpp/oneapi/dal/algo:minkowski_distance",
        "@onedal//cpp/oneapi/dal/algo:chebyshev_distance",
        "@onedal//cpp/oneapi/dal/algo:cosine_distance",
    ],
    extra_deps = [
        "@onedal//cpp/daal/src/algorithms/k_nearest_neighbors:kernel",
    ],
)

dal_test_suite(
    name = "interface_tests",
    framework = "catch2",
    srcs = glob([
        "test/*.cpp",
    ]),
    dal_deps = [
        ":knn",
    ],
)

dal_test_suite(
    name = "tests",
    tests = [
        ":interface_tests",
    ],
)
