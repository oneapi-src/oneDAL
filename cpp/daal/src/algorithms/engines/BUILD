package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:daal.bzl", "daal_module")

daal_module(
    name = "engines",
    hdrs = glob(["**/*.h", "**/*.i"]),
    srcs = glob(["**/*.cpp"]),
    deps = [
        "@onedal//cpp/daal:core",
    ],
)
