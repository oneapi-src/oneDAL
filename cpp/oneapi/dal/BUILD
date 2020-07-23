package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_static_lib",
    "dal_cpu_dispatcher"
)

dal_cpu_dispatcher(
    name = "cpu-dispatcher",
    src = "backend/dispatcher_cpu.hpp",
)

dal_module(
    name = "common",
    dpc = True,
    auto = True,
    includes = [ "../../../cpp" ],
    deps = [
        "@onedal//cpp/daal:all_headers",
        ":cpu-dispatcher",
    ],
)

dal_module(
    name = "core",
    dpc = True,
    deps = [
        ":common",
        "@onedal//cpp/oneapi/dal/data",
        "@onedal//cpp/oneapi/dal/util",
        "@micromkl_dpc//:mkl_dpc",
    ]
)

dal_static_lib(
    name = "static",
    lib_name = "onedal",
    dpc = True,
    deps = [
        ":core",
        "@onedal//cpp/oneapi/dal/algo:decision_forest",
        "@onedal//cpp/oneapi/dal/algo:kmeans",
        "@onedal//cpp/oneapi/dal/algo:kmeans_init",
        "@onedal//cpp/oneapi/dal/algo:knn",
        "@onedal//cpp/oneapi/dal/algo:linear_kernel",
        "@onedal//cpp/oneapi/dal/algo:pca",
        "@onedal//cpp/oneapi/dal/algo:rbf_kernel",
        "@onedal//cpp/oneapi/dal/algo:svm",
        "@micromkl//:mkl_seq",
    ]
)

filegroup(
    name = "static-all",
    srcs = [
        ":static",
        ":static-dpc",
    ],
)
