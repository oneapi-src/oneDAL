package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl",
    "dal_collect_modules",
    "dal_collect_test_suites",
)

ALGOS = [
    "basic_statistics",
    "chebyshev_distance",
    "connected_components",
    "cosine_distance",
    "covariance",
    "decision_forest",
    "decision_tree",
    "jaccard",
    "kmeans",
    "kmeans_init",
    "knn",
    "linear_kernel",
    "linear_regression",
    "louvain",
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
