package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:daal.bzl", "daal_module")

daal_module(
    name = "kernel",
    auto = True,
    opencl = True,
    deps = [
        "@onedal//cpp/daal:core",
        "@onedal//cpp/daal:sycl",
        "@onedal//cpp/daal/src/algorithms/classifier:kernel",
        "@onedal//cpp/daal/src/algorithms/dtrees:kernel",
        "@onedal//cpp/daal/src/algorithms/dtrees/forest:kernel",
    ],
)

daal_module(
    name = "model_impl",
    hdrs = glob(["*.hpp"]),
)
