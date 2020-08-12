load("@onedal//dev/bazel:repos.bzl", "repos")

tbb_repo = repos.prebuilt_libs_repo_rule(
    includes = [
        "include",
    ],
    libs = [
        "lib/intel64/gcc4.8/libtbb.so",
        "lib/intel64/gcc4.8/libtbb.so.12",
        "lib/intel64/gcc4.8/libtbbmalloc.so",
        "lib/intel64/gcc4.8/libtbbmalloc.so.2",
    ],
    build_template = "@onedal//dev/bazel/deps:tbb.tpl.BUILD",
)
