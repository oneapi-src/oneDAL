load("@onedal//dev/bazel:repos.bzl", "repos")

micromkl_repo = repos.prebuilt_libs_repo_rule(
    root_env_var = "MKLFPKROOT",
    url = "https://github.com/oneapi-src/oneDAL/releases/download/Dependencies/mklfpk_lnx_2021.1-beta08.tgz",
    sha256 = "cc1142f0cfd831e394a09231f89946ce84e87b33212845a46d5e869370570962",
    includes = [
        "include",
        "%{os}/include",
    ],
    libs = [
        "%{os}/lib/intel64/libdaal_mkl_thread.a",
        "%{os}/lib/intel64/libdaal_mkl_sequential.a",
        "%{os}/lib/intel64/libdaal_vmlipp_core.a",
    ],
    build_template = "@onedal//dev/bazel/micromkl:micromkl.BUILD.tpl",
)

micromkl_dpc_repo = repos.prebuilt_libs_repo_rule(
    root_env_var = "MKLGPUFPKROOT",
    url = "https://github.com/oneapi-src/oneDAL/releases/download/Dependencies/mklgpufpk_lnx_20200414.tgz",
    includes = [
        "include",
    ],
    libs = [
        "lib/intel64/libdaal_sycl.a",
    ],
    build_template = "@onedal//dev/bazel/micromkl:micromkldpc.BUILD.tpl",
)
