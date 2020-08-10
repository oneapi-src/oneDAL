load("@onedal//dev/bazel:repos.bzl", "repos")

tbb_repo = repos.prebuilt_libs_repo_rule(
    root_env_var = "TBBROOT",
    url = "https://github.com/oneapi-src/oneTBB/releases/download/v2021.1-beta08/oneapi-tbb-2021.1-beta08-lin.tgz",
    sha256 = "02cfd300e3880f2376457feeb8d29e0a3aabf2ac4caf0883b57ca1a2dba073f0",
    strip_prefix = "oneapi-tbb-2021.1-beta08",
    includes = [
        "include",
    ],
    libs = [
        "lib/intel64/gcc4.8/libtbb.so",
        "lib/intel64/gcc4.8/libtbb.so.12",
        "lib/intel64/gcc4.8/libtbbmalloc.so",
        "lib/intel64/gcc4.8/libtbbmalloc.so.2",
    ],
    build_template = "@onedal//dev/bazel/tbb:tbb.BUILD.tpl",
)
