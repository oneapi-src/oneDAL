package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl",
    "dal_module",
)

dal_module(
    name = "common",
    hdrs = [
        "common.hpp",
        "macro.hpp",
        "config.hpp",
    ],
    srcs = [
        "config.cpp",
    ],
    dpc_srcs = [
        "common_dpc.cpp",
    ],
    dal_deps = [
        "@onedal//cpp/oneapi/dal:common",
    ],
    testonly = True,
)

dal_module(
    name = "datasets",
    hdrs = glob(["datasets.hpp"]),
    srcs = glob(["datasets.cpp"]),
    dal_deps = [
        ":common",
        "@onedal//cpp/oneapi/dal:table",
    ],
    testonly = True,
)
