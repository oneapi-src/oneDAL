load("@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_example_suite",
    "dal_algo_example_suite",
)

dal_module(
    name = "example_util",
    hdrs = glob(["source/example_util/*.hpp"]),
    includes = [ "source" ],
)

_TEST_DEPS = [
    ":example_util",
]

_DATA_DEPS = [
    "@onedal//examples/oneapi:data",
]

dal_example_suite(
    name = "graph",
    srcs = glob(["source/graph/*.cpp"]),
    deps = [
        "@onedal//cpp/oneapi/dal/graph",
    ],
    data = _DATA_DEPS,
    test_deps = _TEST_DEPS,
)

dal_example_suite(
    name = "table",
    srcs = glob(["source/table/*.cpp"]),
    deps = [
        "@onedal//cpp/oneapi/dal/table",
    ],
    data = _DATA_DEPS,
    test_deps = _TEST_DEPS,
)

dal_algo_example_suite(
    algos = [
        "decision_forest",
        "jaccard",
        "kmeans",
        "kmeans_init",
        "knn",
        "linear_kernel",
        "pca",
        "rbf_kernel",
        "svm",
    ],
    data = _DATA_DEPS,
    test_deps = _TEST_DEPS,
)
