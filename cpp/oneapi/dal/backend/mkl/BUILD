package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_test_suite",
)

dal_module(
    name = "mkl",
    auto = True,
    dal_deps = [
        "@onedal//cpp/oneapi/dal:common",
    ],
)

dal_test_suite(
    name = "tests",
    dpc = False, # TODO: Remove this once DPC++ support
                 #       is added to `mkl` module

)
