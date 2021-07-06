package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl",
    "dal_collect_modules",
    "dal_collect_test_suites",
)

ALGOS = [
    "chebyshev_distance",
    "cosine_distance",
    "decision_forest",
    "decision_tree",
    "jaccard",
    "kmeans",
    "kmeans_init",
    "knn",
    "linear_kernel",
    "minkowski_distance",
    "pca",
    "polynomial_kernel",
    "rbf_kernel",
    "sigmoid_kernel",
    "shortest_paths",
    "subgraph_isomorphism",
    "svm",
    "triangle_counting",
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
