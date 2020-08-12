load("@onedal//dev/bazel:repos.bzl", "repos")

onedal_repo = repos.prebuilt_libs_repo_rule(
    includes = [
        "include",
    ],
    libs = [
        "lib/intel64/libonedal_core.a",
        "lib/intel64/libonedal_thread.a",
        "lib/intel64/libonedal_sequential.a",
        "lib/intel64/libonedal.a",
        "lib/intel64/libonedal_dpc.a",
        "lib/intel64/libonedal_sycl.a",
    ],
    build_template = "@onedal//dev/bazel/deps:onedal.tpl.BUILD",
)
