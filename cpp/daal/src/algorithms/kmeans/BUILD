package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:daal.bzl", "daal_module")

daal_module(
    name = "kmeans",
    hdrs = glob(["**/*.h", "**/*.i", "**/*.cl"],
                exclude=["**/*init*"]),
    srcs = glob(["**/*.cpp"],
                exclude=["**/*init*"]),
    deps = [
        "@onedal//cpp/daal:core_sycl",
        "@onedal//cpp/daal/src/algorithms/distributions",
    ],
)

daal_module(
    name = "kmeans_init",
    hdrs = glob(["**/*init*.h", "**/*init*.i", "**/*init*.cl"]),
    srcs = glob(["**/*init*.cpp"]),
    deps = [
        "@onedal//cpp/daal:core_sycl",
        "@onedal//cpp/daal/src/algorithms/distributions",
    ],
)

