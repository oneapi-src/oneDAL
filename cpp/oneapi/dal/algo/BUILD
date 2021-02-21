package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl",
    "dal_collect_modules",
    "dal_collect_test_suites",
)

ALGOS = [
    "decision_forest",
    "jaccard",
    "subgraph_isomorphism",
    "kmeans",
    "kmeans_init",
    "knn",
    "linear_kernel",
    "pca",
    "rbf_kernel",
    "svm",
]

dal_collect_modules(
    name = "algo",
    root = "@onedal//cpp/oneapi/dal/algo",
    modules = ALGOS,
)

dal_collect_test_suites(
    name = "tests",
    root = "@onedal//cpp/oneapi/dal/algo",
    modules = ALGOS,
)
