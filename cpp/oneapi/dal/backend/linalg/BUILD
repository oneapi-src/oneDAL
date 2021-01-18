package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_test_suite",
)

dal_module(
    name = "linalg",
    auto = True,
    dal_deps = [
        "@onedal//cpp/oneapi/dal:common",
        "@onedal//cpp/oneapi/dal/table",
        "@onedal//cpp/oneapi/dal/backend/mkl",
    ],
)

dal_test_suite(
    name = "tests",
    framework = "catch2",
    dpc = False, # TODO: Remove this once DPC++ support
                 #       is added to `linalg` module
    srcs = glob(["test/*.cpp"]),
    dal_deps = [
        ":linalg",
    ],
)
