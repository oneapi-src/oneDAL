package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:daal.bzl", "daal_module")

daal_module(
    name = "headers",
    hdrs = glob(["*.h", "*.i"]),
    deps = [
        "@onedal//cpp/daal:core",
    ],
)

daal_module(
    name = "kernel",
    auto = True,
    deps = [
        "@onedal//cpp/daal:core",
        "@onedal//cpp/daal/src/algorithms/normalization/zscore/inner:kernel",
    ],
)
