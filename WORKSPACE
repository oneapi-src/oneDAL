workspace(name = "onedal")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

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
    url = "https://github.com/oneapi-src/oneDAL/releases/download/Dependencies/mklfpk_lnx_20230413.tgz",
    sha256 = "e99dd6fb18f1fda382c53373262d1bb44c1b58aa6edff94cfb0e9d8dcd3395ed",
)

micromkl_dpc_repo(
    name = "micromkl_dpc",
    root_env_var = "MKLGPUFPKROOT",
    url = "https://github.com/oneapi-src/oneDAL/releases/download/Dependencies/mklgpufpk_lnx_2024-02-20.tgz",
    sha256 = "1c60914461aafa5e5512181c7d5c1fdbdeff83746dbd980fe97074a3b65fc1ed",
)

load("@onedal//dev/bazel/deps:openblas.bzl", "openblas_repo")
openblas_repo(
    name = "openblas",
    root_env_var = "OPENBLASROOT",
)

load("@onedal//dev/bazel/deps:tbb.bzl", "tbb_repo")
tbb_repo(
    name = "tbb",
    root_env_var = "TBBROOT",
    url = "https://github.com/oneapi-src/oneTBB/releases/download/v2021.7.0/oneapi-tbb-2021.7.0-lin.tgz",
    sha256 = "3c2b3287c595e2bb833c025fcd271783963b7dfae8dc681440ea6afe5d550e6a",
    strip_prefix = "oneapi-tbb-2021.7.0",
)

load("@onedal//dev/bazel/deps:mpi.bzl", "mpi_repo")
mpi_repo(
    name = "mpi",
    root_env_var = "MPIROOT",
    urls = [
        "https://files.pythonhosted.org/packages/83/3c/c684b721f08f55fc1647d9bcb84e657a4b6217c078a209f2f7751b639957/impi_rt-2021.8.0-py2.py3-none-manylinux1_x86_64.whl",
        "https://files.pythonhosted.org/packages/26/27/5b557da775ad23b20be85352e273dbdddd6ba2a531c6db21e3c9c02230f6/impi_devel-2021.8.0-py2.py3-none-manylinux1_x86_64.whl",
    ],
    sha256s = [
        "c80f86a2a9ff9d4d1b81c1559e9b9180a8c72fb9902ea7d61b07e6d71ad33225",
        "65dfc774e1e853a36c2c6286ea0b8dc33ea1f1f2a6fcd271b917195e11ddc98a",
    ],
    strip_prefixes = [
        "impi_rt-2021.8.0.data/data",
        "impi_devel-2021.8.0.data/data",
    ]
)

load("@onedal//dev/bazel/deps:ccl.bzl", "ccl_repo")
ccl_repo(
    name = "ccl",
    root_env_var = "CCL_ROOT",
    urls = [
        "https://files.pythonhosted.org/packages/38/66/889fa7f40d4142194414efb3f79379ff2830e554d99eef49ac4d8e739245/oneccl_devel-2021.8.0-py2.py3-none-manylinux1_x86_64.whl",
    ],
    sha256s = [
        "58995b5dd38f92034be7c7c2da51f531b6df4ef36b09fb648757ae0a7832f802",
    ],
)

load("@onedal//dev/bazel/deps:mkl.bzl", "mkl_repo")
mkl_repo(
    name = "mkl",
    root_env_var = "MKLROOT",
    urls = [
        "https://files.pythonhosted.org/packages/76/8c/2e6fb6186fa9335a0feb7845e001e18c22627a06ae68650e5a84ca2b536d/mkl_static-2023.0.0-py2.py3-none-manylinux1_x86_64.whl",
        "https://files.pythonhosted.org/packages/cf/d1/ea2d769006337d968a89337dd1c3eb09c528f9ac629e8ab99324e1122f03/mkl_include-2023.0.0-py2.py3-none-manylinux1_x86_64.whl",
    ],
    sha256s = [
        "49d16f315f6803b1046a4796686af766ad487f9f6d98ea76b6cdb2ebd5b559f9",
        "14b0958dff799378975d83fbd00ce756645aa36b9f924bdfdb0fb031f72b734d",
    ],
    strip_prefixes = [
        "mkl_static-2023.0.0.data/data",
        "mkl_include-2023.0.0.data/data",
    ],
)

load("@onedal//dev/bazel/deps:onedal.bzl", "onedal_repo")
onedal_repo(
    name = "onedal_release",
    root_env_var = "DALROOT",
)

http_archive(
    name = "catch2",
    url = "https://github.com/catchorg/Catch2/archive/v3.5.4.tar.gz",
    sha256 = "b7754b711242c167d8f60b890695347f90a1ebc95949a045385114165d606dbb",
    strip_prefix = "Catch2-3.5.4",
)

http_archive(
    name = "fmt",
    url = "https://github.com/fmtlib/fmt/archive/10.2.1.tar.gz",
    sha256 = "1250e4cc58bf06ee631567523f48848dc4596133e163f02615c97f78bab6c811",
    strip_prefix = "fmt-10.2.1",
    build_file = "@onedal//dev/bazel/deps:fmt.tpl.BUILD",
)
