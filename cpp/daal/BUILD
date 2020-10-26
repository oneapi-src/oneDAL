package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:daal.bzl",
    "daal_module",
    "daal_static_lib",
    "daal_dynamic_lib",
    "daal_algorithms",
    "daal_generate_version",
    "daal_patch_kernel_defines",
)

daal_module(
    name = "microvmlipp",
    override_deps_lib_tag = True,
    deps = [
        "@micromkl//:vml_ipp",
        # TODO: Currently vml_ipp lib depends on TBB, but it shouldn't
        #       Remove TBB from deps once problem with vml_ipp is resolved
        "@tbb//:tbb_binary",
    ],
)

daal_module(
    name = "micromkl_sequential",
    override_deps_lib_tag = True,
    deps = [
        "@micromkl//:mkl_seq",
    ],
)

daal_module(
    name = "micromkl_thread",
    override_deps_lib_tag = True,
    deps = [
        "@micromkl//:mkl_thr",
    ],
)

daal_patch_kernel_defines(
    name = "kernel_defines",
    src = "include/services/internal/daal_kernel_defines.h",
)

daal_module(
    name = "public_includes",
    hdrs = glob(["include/**/*.h"], exclude=["**/daal_kernel_defines.h"]),
    includes = [ "include" ],
)

daal_module(
    name = "includes",
    hdrs = [
        ":kernel_defines",
    ],
    includes = [ "." ],
    defines = [
        "DAAL_NOTHROW_EXCEPTIONS",
        "DAAL_HIDE_DEPRECATED",
    ],
    deps = [
        ":public_includes",
    ],
)

daal_generate_version(
    name = "version",
    out = "daal_version_gen.h",
)

daal_module(
    name = "service_headers",
    hdrs = glob([
        "src/services/**/*.h",
        "src/externals/**/*.h",
        "src/algorithms/*.h",
    ]) + [
        ":version",
    ],
    deps = [
        ":includes",
        "@micromkl//:headers",
    ],
)

daal_module(
    name = "threading_headers",
    hdrs = glob(["src/threading/**/*.h"]),
    deps = [
        ":service_headers",
    ],
)

daal_module(
    name = "services",
    srcs = glob([
            "src/services/**/*.cpp",
            "src/externals/**/*.cpp",
            "src/algorithms/*.cpp",
        ],
        exclude = ["src/externals/**/*_win_dll.cpp"],
    ),
    deps = [
        ":service_headers",
        ":threading_headers",
        ":microvmlipp",
    ],
)

daal_module(
    name = "data_management",
    hdrs = glob(["src/data_management/**/*.h"]),
    srcs = glob(["src/data_management/**/*.cpp"]),
    deps = [
        ":services",
    ],
)

daal_module(
    name = "sycl",
    hdrs = glob(["src/sycl/**/*.h", "src/sycl/**/*.cl"]),
    srcs = glob(["src/sycl/**/*.cpp"]),
    deps = [
        ":services",
        "@onedal//cpp/daal/src/algorithms/engines:kernel",
    ],
)

daal_module(
    name = "threading_seq",
    srcs = glob(["src/threading/**/*.cpp"]),
    local_defines = [
        "__DO_SEQ_LAYER__",
    ],
    deps = [
        ":threading_headers",
        ":micromkl_sequential",
    ],
)

daal_module(
    name = "threading_tbb",
    srcs = glob(["src/threading/**/*.cpp"]),
    local_defines = [
        "__DO_TBB_LAYER__",
        "__TBB_NO_IMPLICIT_LINKAGE",
        "__TBB_LEGACY_MODE",
        "TBB_SUPPRESS_DEPRECATED_MESSAGES",
        "TBB_USE_ASSERT=0",
    ],
    deps = [
        ":threading_headers",
        ":micromkl_thread",
        "@tbb//:tbb",
        "@tbb//:tbbmalloc",
    ],
)

daal_module(
    name = "core",
    deps = [
        ":services",
        ":data_management",
    ],
)

daal_algorithms(
    name = "all_algorithms",
    algorithms = [
        "adaboost",
        "assocrules",
        "boosting/inner",
        "brownboost",
        "cholesky",
        "classifier",
        "cordistance",
        "cosdistance",
        "covariance",
        "dbscan",
        "decision_tree",
        "distributions",
        "dtrees",
        "dtrees/forest/classification",
        "dtrees/forest/regression",
        "dtrees/gbt/regression",
        "dtrees/gbt/classification",
        "elastic_net",
        "em",
        "engines",
        "implicit_als",
        "k_nearest_neighbors",
        "kernel_function",
        "kmeans",
        "lasso_regression",
        "linear_model",
        "linear_regression",
        "logistic_regression",
        "logitboost",
        "low_order_moments",
        "multiclassclassifier",
        "naivebayes",
        "normalization/minmax",
        "normalization/zscore",
        "objective_function",
        "objective_function/cross_entropy_loss",
        "objective_function/logistic_loss",
        "objective_function/mse",
        "optimization_solver",
        "optimization_solver/sgd",
        "optimization_solver/lbfgs",
        "optimization_solver/adagrad",
        "optimization_solver/saga",
        "optimization_solver/coordinate_descent",
        "outlierdetection_bacon",
        "outlierdetection_multivariate",
        "outlierdetection_univariate",
        "pca",
        "pca/metrics",
        "pca/transform",
        "pivoted_qr",
        "qr",
        "quantiles",
        "regression",
        "ridge_regression",
        "sorting",
        "stump",
        "svd",
        "svm",
        "weak_learner/inner",
    ],
)

daal_static_lib(
    name = "core_static",
    lib_name = "onedal_core",
    deps = [
        ":core",
        ":all_algorithms",
    ],
)

daal_static_lib(
    name = "sequential_static",
    lib_name = "onedal_sequential",
    deps = [
        ":threading_seq",
    ],
)

daal_static_lib(
    name = "thread_static",
    lib_name = "onedal_thread",
    deps = [
        ":threading_tbb",
    ],
)

daal_dynamic_lib(
    name = "core_dynamic",
    lib_name = "onedal_core",
    deps = [
        ":core",
        ":all_algorithms",
    ],
)

daal_dynamic_lib(
    name = "sequential_dynamic",
    lib_name = "onedal_sequential",
    deps = [
        ":threading_seq",
    ],
    def_file = "src/threading/export_lnx32e.def",
)

daal_dynamic_lib(
    name = "thread_dynamic",
    lib_name = "onedal_thread",
    deps = [
        ":threading_tbb",
    ],
    def_file = "src/threading/export_lnx32e.def",
)

daal_module(
    name = "threading_static",
    deps = select({
        "@config//:par_test_thread_mode": [
            "@onedal//cpp/daal:thread_static",
        ],
        "@config//:seq_test_thread_mode": [
            "@onedal//cpp/daal:sequential_static",
        ],
    }),
)

daal_module(
    name = "threading_dynamic",
    deps = select({
        "@config//:par_test_thread_mode": [
            "@onedal//cpp/daal:thread_dynamic",
        ],
        "@config//:seq_test_thread_mode": [
            "@onedal//cpp/daal:sequential_dynamic",
        ],
    }),
)

daal_module(
    name = "threading_release_static",
    deps = select({
        "@config//:par_test_thread_mode": [
            "@onedal_release//:thread_static",
        ],
        "@config//:seq_test_thread_mode": [
            "@onedal_release//:sequential_static",
        ],
    }),
)

daal_module(
    name = "threading_release_dynamic",
    deps = select({
        "@config//:par_test_thread_mode": [
            "@onedal_release//:thread_dynamic",
        ],
        "@config//:seq_test_thread_mode": [
            "@onedal_release//:sequential_dynamic",
        ],
    }),
)

filegroup(
    name = "all_static",
    srcs = [
        ":core_static",
        ":sequential_static",
        ":thread_static",
    ],
)
