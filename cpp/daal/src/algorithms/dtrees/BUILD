package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:daal.bzl", "daal_module")

daal_module(
    name = "kernel",
    hdrs = glob(["**/*.h", "**/*.i", "**/*.cl"]),
    srcs = glob(["*.cpp"]),
    deps = [
        "@onedal//cpp/daal:core",
    ],
)
