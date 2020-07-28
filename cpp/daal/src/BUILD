package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:daal.bzl", "daal_module")

daal_module(
    name = "service_headers",
    hdrs = glob([
        "services/**/*.h",
        "externals/**/*.h",
        "algorithms/service_*.h",
    ]),
    deps = [
        "@onedal//cpp/daal:includes",
    ],
)

daal_module(
    name = "threading_seq",
    lib_tag = "daal_sequential",
    hdrs = glob(["threading/*.h"]),
    srcs = glob(["threading/*.cpp"]),
    local_defines = [
        "__DO_SEQ_LAYER__",
    ],
    deps = [
        ":service_headers",
    ],
)

daal_module(
    name = "threading_tbb",
    lib_tag = "daal_sequential",
    hdrs = glob(["threading/*.h"]),
    srcs = glob(["threading/*.cpp"]),
    local_defines = [
        "__DO_TBB_LAYER__",
        "TBB_USE_ASSERT=0",
    ],
    deps = [
        ":service_headers",
    ],
)

daal_module(
    name = "services",
    lib_tag = "daal_core",
    srcs = glob(
        ["services/**/*.cpp", "externals/**/*.cpp"],
        exclude = ["externals/**/*_win_dll.cpp"],
    ),
    deps = [
        ":service_headers",
        ":threading_seq",
        "@config//:daal_version_data",
        "@micromkl//:vml_ipp",
    ],
)

daal_module(
    name = "data_management",
    lib_tag = "daal_core",
    hdrs = glob(["data_management/**/*.h"]),
    srcs = glob(["data_management/**/*.cpp"]),
    deps = [
        ":services",
    ],
)
