load("@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_algo_example_suite",
)

_DATA_DEPS = [
    "@onedal//examples/daal:data_daal",
]

dal_module(
    name = "utils",
    hdrs = glob(["source/utils/*.h"]),
    includes = [ "source",
    "source/utils" ],
)

_TEST_DEPS = [
    ":utils",
]

# dal_example_suite(
#     name = "graph",
#     compile_as = [ "c++" ],
#     srcs = glob(["source/graph/*.cpp"]),
#     dal_deps = [
#         "@onedal//cpp/oneapi/dal/graph",
#     ],
#     data = _DATA_DEPS,
#     extra_deps = _TEST_DEPS,
# )

# dal_example_suite(
#     name = "table",
#     compile_as = [ "c++" ],
#     srcs = glob(["source/table/*.cpp"]),
#     dal_deps = [
#         "@onedal//cpp/oneapi/dal/table",
#     ],
#     data = _DATA_DEPS,
#     extra_deps = _TEST_DEPS,
# )

# dal_example_suite(
#     name = "kmeans_init",
#     compile_as = [ "c++" ],
#     srcs = glob(["source/kmeans_init/*.cpp"]),
#     dal_deps = [
#         "@onedal//cpp/oneapi/dal/algo:kmeans",
#         "@onedal//cpp/oneapi/dal/algo:kmeans_init",
#     ],
#     data = _DATA_DEPS,
#     extra_deps = _TEST_DEPS,
# )

# dal_example_suite(
#     name = "misc",
#     compile_as = [ "c++" ],
#     srcs = glob(["source/misc/*.cpp"]),
#     dal_deps = [
#         "@onedal//cpp/oneapi/dal:io",
#     ],
#     data = _DATA_DEPS,
#     extra_deps = _TEST_DEPS,
# )

# dal_example_suite(
#     name = "jaccard",
#     compile_as = [ "c++" ],
#     srcs = glob(["source/jaccard/*.cpp"]),
#     dal_deps = [
#         "@onedal//cpp/oneapi/dal/algo:jaccard",
#     ],
#     data = _DATA_DEPS,
#     extra_deps = _TEST_DEPS + [
#         "@tbb//:tbb",
#     ],
# )

dal_algo_example_suite(
    algos = [
        "kmeans",
    ],
    compile_as = [ "c++" ],
    data = _DATA_DEPS,
    extra_deps = _TEST_DEPS,
)
