package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl", "dal_module")

dal_module(
    name = "kmeans",
    dpc = True,
    auto = True,
    deps = [
        "@onedal//cpp/oneapi/dal:core",
        "@onedal//cpp/daal/src/algorithms/kmeans:kmeans",
        "@onedal//cpp/daal/src/algorithms/kmeans:kmeans_init",
    ]
)
