workspace(name = "onedal")

load("@onedal//dev/bazel/toolchains:configure_auto.bzl",
    "declare_onedal_cc_toolchain"
)
declare_onedal_cc_toolchain(
    name = "onedal_cc_toolchain",
)

load("@onedal//dev/bazel/config:def.bzl",
    "declare_onedal_config"
)
declare_onedal_config(
    name = "config",
)

load("@onedal//dev/bazel/micromkl:micromkl.bzl",
    "micromkl_repo",
    "micromkl_dpc_repo"
)
micromkl_repo(
    name = "micromkl",
)
micromkl_dpc_repo(
    name = "micromkl_dpc",
)

load("@bazel_tools//tools/build_defs/repo:http.bzl",
    "http_archive"
)
http_archive(
  name = "bazel_skylib",
  urls = [
    "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.0.2/bazel-skylib-1.0.2.tar.gz",
    "https://github.com/bazelbuild/bazel-skylib/releases/download/1.0.2/bazel-skylib-1.0.2.tar.gz",
  ],
  sha256 = "97e70364e9249702246c0e9444bccdc4b847bed1eb03c5a3ece4f83dfe6abc44",
)

http_archive(
    name = "gtest",
    url = "https://github.com/google/googletest/archive/release-1.10.0.tar.gz",
    sha256 = "9dc9157a9a1551ec7a7e43daea9a694a0bb5fb8bec81235d8a1e6ef64c716dcb",
    strip_prefix = "googletest-release-1.10.0",
)
