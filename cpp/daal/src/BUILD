package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:cc.bzl", "cc_module")

cc_module(
    name = "services",
    hdrs = glob(["services/**/*.h", "externals/**/*.h"]),
    srcs = glob(
        ["services/**/*.cpp", "externals/**/*.cpp"],
        exclude = ["externals/**/*_win_dll.cpp"]
    ),
    deps = [
        "@onedal//cpp/daal:includes",
        "@micromkl//:vml_ipp",
        "@config//:daal_version_data",
    ],
)

cc_module(
    name = "data_management",
    hdrs = glob(["data_management/**/*.h"]),
    srcs = glob(["data_management/**/*.cpp"]),
    deps = [
        "@onedal//cpp/daal:includes",
        ":services",
    ],
)

