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
    ]
)

dal_test_suite(
    name = "tests",
    framework = "catch2",
    srcs = glob([
        "test/train.cpp",
    ]),
    dal_deps = [
        ":pca",
        "@onedal//cpp/oneapi/dal/test:datasets",
        "@onedal//cpp/oneapi/dal/backend/linalg",
    ],
)
