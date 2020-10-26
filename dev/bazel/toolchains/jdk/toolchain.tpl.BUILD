package(default_visibility = ["//visibility:public"])

load("@onedal//dev/bazel/toolchains/jdk:toolchain_config.bzl", "extra_jdk_tools")

alias(
    name = "vanilla",
    actual = "@bazel_tools//tools/jdk:toolchain_vanilla"
)

java_runtime(
    name = "local",
    java_home = "%{java_home}",
)

cc_library(
    name = "jni_headers",
    hdrs = glob(["include/**/*.h"]),
    includes = ["include", "include/linux"],
)

extra_jdk_tools(
    name = "extra_jdk_tools",
    extract_jni_headers = "%{extract_jni_headers}",
)

toolchain(
    name = "extra_jdk_tools_lnx",
    exec_compatible_with = [
        "@platforms//cpu:x86_64",
        "@platforms//os:linux",
    ],
    target_compatible_with = [
        "@platforms//cpu:x86_64",
        "@platforms//os:linux",
    ],
    toolchain = ":extra_jdk_tools",
    toolchain_type = "@onedal//dev/bazel/toolchains/jdk:extra_tools",
)
