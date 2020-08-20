package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_collect_modules",
    "dal_public_includes",
    "dal_static_lib",
    "dal_test_suite",
    "dal_generate_cpu_dispatcher",
)

dal_generate_cpu_dispatcher(
    name = "cpu_dispatcher",
    out = "_dal_cpu_dispatcher_gen.hpp",
)

dal_module(
    name = "common",
    auto = True,
    includes = [ ".", "../.." ],
    system_includes = [
        "../../daal",
        "../../daal/include",
    ],
    hdrs = [
        ":cpu_dispatcher",
    ],
    extra_deps = [
        "@onedal//cpp/daal:services",
        "@onedal//cpp/daal:data_management",
    ],
)

dal_collect_modules(
    name = "core",
    root = "@onedal//cpp/oneapi/dal",
    modules = [
        "graph",
        "table",
        "util",
    ],
    dal_deps = [
        ":common",
    ],
    dpc_deps = [
        "@micromkl_dpc//:mkl_dpc",
    ],
)

dal_collect_modules(
    name = "optional",
    root = "@onedal//cpp/oneapi/dal",
    modules = [
        "algo",
        "io",
    ],
)

dal_public_includes(
    name = "public_includes",
    dal_deps = [
        ":core",
        ":optional",
    ],
)

dal_static_lib(
    name = "static",
    lib_name = "onedal",
    dal_deps = [
        ":core",
        ":optional",
    ],
    extra_deps = [
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
    dal_deps = [ ":common" ],
)
