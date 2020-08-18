package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl", "dal_module", "dal_public_includes")

dal_public_includes(
    name = "public_includes",
)

dal_module(
    name = "knn",
    dpc = True,
    auto = True,
    deps = [
        "@onedal//cpp/oneapi/dal:core",
        "@onedal//cpp/daal/src/algorithms/k_nearest_neighbors:kernel",
    ]
)
