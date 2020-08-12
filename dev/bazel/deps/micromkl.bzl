load("@onedal//dev/bazel:repos.bzl", "repos")

micromkl_repo = repos.prebuilt_libs_repo_rule(
    includes = [
        "include",
        "%{os}/include",
    ],
    libs = [
        "%{os}/lib/intel64/libdaal_mkl_thread.a",
        "%{os}/lib/intel64/libdaal_mkl_sequential.a",
        "%{os}/lib/intel64/libdaal_vmlipp_core.a",
    ],
    build_template = "@onedal//dev/bazel/deps:micromkl.tpl.BUILD",
)

micromkl_dpc_repo = repos.prebuilt_libs_repo_rule(
    includes = [
        "include",
    ],
    libs = [
        "lib/intel64/libdaal_sycl.a",
    ],
    build_template = "@onedal//dev/bazel/deps:micromkldpc.tpl.BUILD",
)
