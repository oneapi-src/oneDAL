package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_static_lib",
    "dal_generate_cpu_dispatcher",
    "dal_algos",
)

dal_generate_cpu_dispatcher(
    name = "cpu_dispatcher",
    out = "_dal_cpu_dispatcher_gen.hpp",
)

dal_module(
    name = "common",
    dpc = True,
    auto = True,
    includes = [ ".", "../.." ],
    system_includes = [ "../../daal" ],
    hdrs = [
        ":cpu_dispatcher",
    ],
    deps = [
        "@onedal//cpp/daal:services",
        "@onedal//cpp/daal:data_management",
    ],
)

dal_module(
    name = "core",
    dpc = True,
    deps = [
        ":common",
        "@onedal//cpp/oneapi/dal/data",
        "@onedal//cpp/oneapi/dal/util",
        # TODO: Move to depc_dpc
        # "@micromkl_dpc//:mkl_dpc",
    ],
)

dal_algos(
    name = "all_algos",
    algos = [
        "decision_forest",
        "kmeans",
        "kmeans_init",
        "knn",
        "linear_kernel",
        "pca",
        "rbf_kernel",
        "svm",
    ]
)

dal_static_lib(
    name = "static",
    lib_name = "onedal",
    dpc = True,
    deps = [
        ":core",
        ":all_algos",
    ],
    external_deps = [
        "@micromkl//:mkl_seq",
    ],
)

filegroup(
    name = "all_static",
    srcs = [
        ":static",
        ":static_dpc",
    ],
)
