load(
    "@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_test_suite",
)

package(default_visibility = ["//visibility:public"])

dal_module(
    name = "spmd_comm",
    hdrs = [
        "spmd_communicator.hpp",
    ],
    dal_deps = [
        "@onedal//cpp/oneapi/dal:common",
    ],
)

dal_module(
    name = "mpi_comm",
    hdrs = [
        "mpi_communicator.hpp",
    ],
    extra_deps = [
        "@mpi//:mpi",
    ],
    dal_deps = [
        ":spmd_comm",
    ],
)

dal_test_suite(
    name = "mpi_comm_test",
    srcs = [
        "test/mpi_communicator.cpp",
    ],
    dal_deps = [":mpi_comm"],
    framework = "catch2",
)

dal_test_suite(
    name = "tests",
    tests = [
        ":mpi_comm_test",
    ],
)
