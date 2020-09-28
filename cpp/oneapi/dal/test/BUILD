package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl",
    "dal_module",
)

dal_module(
    name = "catch2_main",
    dpc = False,
    srcs = [
        "main.cpp"
    ],
    extra_deps = [
        "@catch2//:catch2"
    ],
)

dal_module(
    name = "common",
    hdrs = glob([
        "common.hpp",
        "macro.hpp"
    ]),
    dal_deps = [
        "@onedal//cpp/oneapi/dal:common",
    ],
    extra_deps = [
        "@catch2//:catch2",
        "@fmt//:fmt",
    ],
)

dal_module(
    name = "datasets",
    hdrs = glob(["datasets.hpp"]),
    srcs = glob(["datasets.cpp"]),
    dal_deps = [
        ":common",
        "@onedal//cpp/oneapi/dal:table",
    ],
)
