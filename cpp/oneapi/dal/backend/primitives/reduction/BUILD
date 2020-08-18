package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl", "dal_module")

dal_module(
    name = "reduction",
    dpc = True,
    auto = True,
    deps = [
        "@onedal//cpp/oneapi/dal:common",
        "@onedal//cpp/oneapi/dal/backend/primitives:common",
    ],
)
