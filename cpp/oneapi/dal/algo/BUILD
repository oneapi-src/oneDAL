package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl",
    "dal_collect_modules",
    "dal_collect_tests",
)

ALGOS = [
    "decision_forest",
    "jaccard",
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

dal_collect_tests(
    name = "tests",
    root = "@onedal//cpp/oneapi/dal/algo",
    modules = ALGOS,
)
