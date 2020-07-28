load("@onedal//dev/bazel:repos.bzl", "repos")

tbb_repo = repos.prebuilt_libs_repo_rule(
    root_env_var = "TBBROOT",
    includes = [
        "include",
    ],
    libs = [
        "lib/intel64/gcc4.8/libtbb.so",
        "lib/intel64/gcc4.8/libtbbmalloc.so",
    ],
    build_template = "@onedal//dev/bazel/tbb:tbb.BUILD.tpl",
)
