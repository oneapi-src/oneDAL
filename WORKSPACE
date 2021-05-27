workspace(name = "onedal")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
  name = "bazel_skylib",
  urls = [
    "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.0.3/bazel-skylib-1.0.3.tar.gz",
    "https://github.com/bazelbuild/bazel-skylib/releases/download/1.0.3/bazel-skylib-1.0.3.tar.gz",
  ],
  sha256 = "1c531376ac7e5a180e0237938a2536de0c54d93f5c278634818e0efc952dd56c",
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
    url = "https://files.pythonhosted.org/packages/13/9b/9122cd616c62f50aeb1c9aa6b118043764bf1468940726e284a81c6013bc/impi_rt-2021.2.0-py2.py3-none-manylinux1_x86_64.whl",
    sha256 = "TODO",
    strip_prefix = "impi_rt-2021.2.0.data/data",
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
    url = "https://github.com/catchorg/Catch2/archive/v2.13.6.tar.gz",
    sha256 = "48dfbb77b9193653e4e72df9633d2e0383b9b625a47060759668480fdf24fbd4",
    strip_prefix = "Catch2-2.13.6",
)

http_archive(
    name = "fmt",
    url = "https://github.com/fmtlib/fmt/archive/7.1.3.tar.gz",
    sha256 = "5cae7072042b3043e12d53d50ef404bbb76949dad1de368d7f993a15c8c05ecc",
    strip_prefix = "fmt-7.1.3",
    build_file = "@onedal//dev/bazel/deps:fmt.tpl.BUILD",
)
