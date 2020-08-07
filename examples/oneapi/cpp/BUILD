load("@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_test",
)

dal_module(
    name = "example_util",
    hdrs = glob(["source/example_util/*.hpp"]),
    includes = [ "source" ],
)

dal_test(
    name = "kmeans",
    srcs = [ "source/kmeans/kmeans_lloyd_dense_batch.cpp" ],
    deps = [
        ":example_util",
        "@onedal//cpp/oneapi/dal/algo:kmeans",
    ],
)
