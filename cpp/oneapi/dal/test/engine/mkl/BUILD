package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl",
    "dal_test_module",
    "dal_test_suite",
)

dal_test_module(
    name = "mkl",
    auto = True,
    dal_test_deps = [
        "@onedal//cpp/oneapi:include_root",
    ],
    extra_deps = [
        "@mkl//:mkl_seq",
    ],
)

dal_test_suite(
    name = "tests",
    dpc = False, # TODO: Remove this once DPC++ support
                 #       is added to `mkl` module

)
