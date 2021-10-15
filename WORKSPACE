workspace(name = "onedal")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
  name = "bazel_skylib",
  urls = [
    "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.1.1/bazel-skylib-1.1.1.tar.gz",
    "https://github.com/bazelbuild/bazel-skylib/releases/download/1.1.1/bazel-skylib-1.1.1.tar.gz",
  ],
  sha256 = "c6966ec828da198c5d9adbaa94c05e3a1c7f21bd012a0b29ba8ddbccb2c93b0d",
)

load("@onedal//dev/bazel/config:config.bzl", "declare_onedal_config")
declare_onedal_config(
    name = "config",
)

load("@onedal//dev/bazel/toolchains:cc_toolchain.bzl", "declare_onedal_cc_toolchain")
declare_onedal_cc_toolchain(
    name = "onedal_cc_toolchain",
)

load("@onedal//dev/bazel/toolchains:extra_toolchain.bzl", "declare_onedal_extra_toolchain")
declare_onedal_extra_toolchain(
    name = "onedal_extra_toolchain",
)

load("@onedal//dev/bazel/deps:opencl.bzl", "opencl_repo")
opencl_repo(
    name = "opencl",
)

load("@onedal//dev/bazel/deps:micromkl.bzl", "micromkl_repo", "micromkl_dpc_repo")
micromkl_repo(
    name = "micromkl",
    root_env_var = "MKLFPKROOT",
    url = "https://github.com/oneapi-src/oneDAL/releases/download/Dependencies/mklfpk_lnx_20210426.tgz",
    sha256 = "5f87df20db0923f7e1819e2fcfaf60dd749408dca8a094e6713f275fb15ad77b",
)

micromkl_dpc_repo(
    name = "micromkl_dpc",
    root_env_var = "MKLGPUFPKROOT",
    url = "https://github.com/oneapi-src/oneDAL/releases/download/Dependencies/mklgpufpk_lnx_20210406.tgz",
    sha256 = "f19b7390dad523f3306ec47bb69350af32e4813dea19b2e7a68fa88861f655a3",
)

load("@onedal//dev/bazel/deps:tbb.bzl", "tbb_repo")
tbb_repo(
    name = "tbb",
    root_env_var = "TBBROOT",
    url = "https://github.com/oneapi-src/oneTBB/releases/download/v2021.2.0/oneapi-tbb-2021.2.0-lin.tgz",
    sha256 = "bcaa0aa7c0d2851a730cb0f64ad120f06a7b63811c8cc49e51ea8e1579f51b05",
    strip_prefix = "oneapi-tbb-2021.2.0",
)

load("@onedal//dev/bazel/deps:mpi.bzl", "mpi_repo")
mpi_repo(
    name = "mpi",
    root_env_var = "MPIROOT",
    urls = [
        "https://files.pythonhosted.org/packages/13/9b/9122cd616c62f50aeb1c9aa6b118043764bf1468940726e284a81c6013bc/impi_rt-2021.2.0-py2.py3-none-manylinux1_x86_64.whl",
        "https://files.pythonhosted.org/packages/b9/9a/f9b0b0af026cc9a63b9ad2ab7da259adef2989dfe76805eb4d0c70422131/impi_devel-2021.3.1-py2.py3-none-manylinux1_x86_64.whl",
    ],
    sha256s = [
        "b52d4dcc8f4bea30c8373676180723ad146a6d80fe92f228c45e1a8d1fe66091",
        "5375a54166baa675b6ee0c9b7b7d9eecfd2f23da258ee4a4f34dd711ac2c5c38",
    ],
    strip_prefixes = [
        "impi_rt-2021.2.0.data/data",
        "impi_devel-2021.3.1.data/data",
    ]
)

load("@onedal//dev/bazel/deps:ccl.bzl", "ccl_repo")
ccl_repo(
    name = "ccl",
    root_env_var = "CCL_ROOT",
    urls = [
        "https://files.pythonhosted.org/packages/ea/d9/3cb54d9b31aea0db527fc3480fc2e92438602ed53de3e45993c0643cf68f/oneccl_devel-2021.3.0-py2.py3-none-manylinux1_x86_64.whl",
    ],
    sha256s = [
        "d6cf242f7c6d4bc86cf6ec38c416b5ad569d3a0bd2d5258ae5e271027a7e6518",
    ],
)

load("@onedal//dev/bazel/deps:mkl.bzl", "mkl_repo")
mkl_repo(
    name = "mkl",
    root_env_var = "MKLROOT",
    urls = [
        "https://files.pythonhosted.org/packages/bd/74/556cd5efce782ebee2832bd29a49426e88caf2e3cfae38e1d23c0abd41d7/mkl_static-2021.1.1-py2.py3-none-manylinux1_x86_64.whl",
        "https://files.pythonhosted.org/packages/7d/70/86d0db59598c34a5c1d334ba996271dddad89108127b743c84beb6354afd/mkl_include-2021.1.1-py2.py3-none-manylinux1_x86_64.whl",
    ],
    sha256s = [
        "1db75a53f58cad32935bfbd63206124f6218d71193eea1e75a16aadb9c078370",
        "865c884473b0a76da201fe972e68c3b2591e6580753485548acc32169db3ffe7",
    ],
    strip_prefixes = [
        "mkl_static-2021.1.1.data/data",
        "mkl_include-2021.1.1.data/data",
    ],
)

load("@onedal//dev/bazel/deps:onedal.bzl", "onedal_repo")
onedal_repo(
    name = "onedal_release",
    root_env_var = "DAALROOT",
)

http_archive(
    name = "gtest",
    url = "https://github.com/google/googletest/archive/release-1.10.0.tar.gz",
    sha256 = "9dc9157a9a1551ec7a7e43daea9a694a0bb5fb8bec81235d8a1e6ef64c716dcb",
    strip_prefix = "googletest-release-1.10.0",
)

http_archive(
    name = "catch2",
    url = "https://github.com/catchorg/Catch2/archive/v2.13.7.tar.gz",
    sha256 = "3cdb4138a072e4c0290034fe22d9f0a80d3bcfb8d7a8a5c49ad75d3a5da24fae",
    strip_prefix = "Catch2-2.13.7",
)

http_archive(
    name = "fmt",
    url = "https://github.com/fmtlib/fmt/archive/8.0.1.tar.gz",
    sha256 = "b06ca3130158c625848f3fb7418f235155a4d389b2abc3a6245fb01cb0eb1e01",
    strip_prefix = "fmt-8.0.1",
    build_file = "@onedal//dev/bazel/deps:fmt.tpl.BUILD",
)
