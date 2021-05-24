load(
    "@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_test_suite",
)

package(default_visibility = ["//visibility:public"])

dal_module(
    name = "common",
    hdrs = [
        "common.hpp",
    ],
    dal_deps = [
        "@onedal//cpp/oneapi/dal:common",
    ],
)

dal_module(
    name = "host",
    hdrs = [
        "host.hpp",
    ],
    dal_deps = [
        ":common",
    ],
)

dal_module(
    name = "sycl",
    hdrs = [
        "sycl.hpp",
    ],
    dal_deps = [
        ":common",
    ],
)

dal_module(
    name = "mpi",
    hdrs = [
        "mpi.hpp",
    ],
    dal_deps = [
        ":host",
        ":sycl",
    ],
    extra_deps = [
        "@mpi//:mpi",
    ],
)

dal_module(
    name = "policy",
    dal_deps = [
        ":host",
        ":sycl",
        # ":mpi" is not included here intentionally to not create
        # dependency on MPI in the modules that depends on policy
    ],
)

dal_test_suite(
    name = "mpi_policy_tests",
    srcs = [
        "test/mpi.cpp",
    ],
    dal_deps = [":mpi"],
    framework = "catch2",
)

dal_test_suite(
    name = "tests",
    tests = [
        ":mpi_policy_tests",
    ],
)
