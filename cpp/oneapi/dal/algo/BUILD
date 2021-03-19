package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl",
    "dal_collect_modules",
    "dal_collect_test_suites",
)

ALGOS = [
    "decision_forest",
    "jaccard",
    "kmeans",
    "kmeans_init",
    "knn",
    "linear_kernel",
    "pca",
    "polynomial_kernel",
    "rbf_kernel",
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
