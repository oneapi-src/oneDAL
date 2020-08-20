package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_static_lib",
    "dal_public_includes",
    "dal_generate_cpu_dispatcher",
    "dal_algos",
    "dal_test_suite",
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
    system_includes = [
        "../../daal",
        "../../daal/include",
    ],
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
        "@onedal//cpp/oneapi/dal/graph",
        "@onedal//cpp/oneapi/dal/table",
        "@onedal//cpp/oneapi/dal/util",
        # TODO: Move to depc_dpc
        # "@micromkl_dpc//:mkl_dpc",
    ],
)

dal_algos(
    name = "all_algos",
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
    ]
)

dal_public_includes(
    name = "public_includes",
    deps = [
        ":all_algos_public_includes",
        "@onedal//cpp/oneapi/dal/graph:public_includes",
        "@onedal//cpp/oneapi/dal/table:public_includes",
        "@onedal//cpp/oneapi/dal/util:public_includes",
        "@onedal//cpp/oneapi/dal/io:public_includes",
    ],
)

dal_static_lib(
    name = "static",
    lib_name = "onedal",
    dpc = True,
    deps = [
        ":core",
        ":all_algos",
        "@onedal//cpp/oneapi/dal/io",
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

dal_test_suite(
    name = "common_tests",
    srcs = [
        "array_test.cpp",
    ],
    deps = [ ":common" ],
)
