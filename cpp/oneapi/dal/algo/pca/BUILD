load("@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_test_suite",
)

package(default_visibility = ["//visibility:public"])

dal_module(
    name = "core",
    auto = True,
    dal_deps = [
        "@onedal//cpp/oneapi/dal:core",
    ],
)

dal_module(
    name = "parameters",
    dal_deps = [
        "@onedal//cpp/oneapi/dal/algo/pca/parameters",
    ],
)

dal_module(
    name = "pca",
    dal_deps = [
        ":core",
        ":parameters",
        "@onedal//cpp/oneapi/dal/algo/pca/detail",
        "@onedal//cpp/oneapi/dal/algo/pca/backend",
    ]
)

dal_test_suite(
    name = "cpu_tests",
    srcs = glob([
        "backend/cpu/test/*.cpp",
    ]),
    compile_as = ["c++"],
    dal_deps = [
        ":pca",
    ],
    private = True,
)

dal_test_suite(
    name = "gpu_tests",
    srcs = glob([
        "backend/gpu/test/*.cpp",
    ]),
    compile_as = ["dpc++"],
    dal_deps = [
        ":pca",
    ],
    private = True,
)

dal_test_suite(
    name = "backend_tests",
    srcs = glob([
        "backend/test/*.cpp",
    ]),
    dal_deps = [
        ":pca",
    ],
    private = True,
    framework = "catch2",
)

dal_test_suite(
    name = "interface_tests",
    srcs = glob([
        "test/*.cpp",
    ]),
    hdrs = glob([
    "test/*.hpp",
    ]),
    dal_deps = [
        ":pca",
    ],
    framework = "catch2",
)

dal_test_suite(
    name = "tests",
    tests = [
        ":cpu_tests",
        ":gpu_tests",
        ":backend_tests",
        ":interface_tests",
    ],
)
