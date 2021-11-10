package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_test_suite",
)

dal_module(
    name = "louvain",
    auto = True,
    dal_deps = [
        "@onedal//cpp/oneapi/dal:core",
        "@onedal//cpp/oneapi/dal/backend/primitives:rng",
    ],
)

dal_test_suite(
    name = "tests",
    framework = "catch2",
    compile_as = [ "c++" ],
    srcs = glob([
        "test/*.cpp",
    ]),
    dal_deps = [
        ":louvain",
    ],
)
