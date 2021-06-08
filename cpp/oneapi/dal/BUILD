package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_collect_modules",
    "dal_public_includes",
    "dal_static_lib",
    "dal_dynamic_lib",
    "dal_test_suite",
    "dal_collect_test_suites",
    "dal_generate_cpu_dispatcher",
)

dal_generate_cpu_dispatcher(
    name = "cpu_dispatcher",
    out = "_dal_cpu_dispatcher_gen.hpp",
)

dal_module(
    name = "common",
    auto = True,
    hdrs = [
        ":cpu_dispatcher",
    ],
    dal_deps = [
        "@onedal//cpp/oneapi:dal_header",
    ],
    extra_deps = [
        "@onedal//cpp/oneapi:include_root",
        "@onedal//cpp/daal:services",
        "@onedal//cpp/daal:data_management",
    ],
    dpc_deps = [
        "@micromkl_dpc//:mkl_dpc",
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
)

dal_collect_modules(
    name = "optional",
    root = "@onedal//cpp/oneapi/dal",
    modules = [
        "algo",
        "io",
        "policy",
        "backend/micromkl",
        "backend/primitives",
    ],
)

dal_public_includes(
    name = "public_includes",
    dal_deps = [
        ":core",
        ":optional",
        # MPI module is not included to policy module to not
        # create dependency on MPI at the library build stage.
        # MPI deps are header-only, so we include them explicitly
        # to public_includes
        "@onedal//cpp/oneapi/dal/policy:mpi",
    ],
)

dal_static_lib(
    name = "static",
    lib_name = "onedal",
    dal_deps = [
        ":core",
        ":optional",
    ],
)

dal_dynamic_lib(
    name = "dynamic",
    lib_name = "onedal",
    dal_deps = [
        ":core",
        ":optional",
    ],
)

filegroup(
    name = "all_static",
    srcs = [
        ":static",
        ":static_dpc",
    ],
)

filegroup(
    name = "all_dynamic",
    srcs = [
        ":dynamic",
        ":dynamic_dpc",
    ],
)

dal_test_suite(
    name = "common_tests",
    framework = "catch2",
    srcs = glob([
        "test/*.cpp",
    ]),
    dal_deps = [ ":common" ],
)

# TODO: Consider backend as a separate module
dal_test_suite(
    name = "backend_tests",
    framework = "catch2",
    srcs = glob([
        "backend/test/*.cpp",
    ]),
    dal_deps = [ ":common" ],
    private = True,
)

dal_collect_test_suites(
    name = "tests",
    root = "@onedal//cpp/oneapi/dal",
    modules = [
        "algo",
        "graph",
        "io",
        "policy",
        "table",
        "util",
        "backend/micromkl",
        "backend/primitives",
    ],
    tests = [
        ":common_tests",
        ":backend_tests",
        # TODO: Temporary disabled due to
        #       unexpectedly high build time
        # "@onedal//cpp/oneapi:dal_hpp_test",
    ],
)
