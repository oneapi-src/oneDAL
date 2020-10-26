package(default_visibility = ["//visibility:public"])

load("@onedal//dev/bazel/toolchains/extra:toolchain_lnx_config.bzl", "extra_toolchain")

extra_toolchain(
    name = "extra_tools",
    patch_daal_kernel_defines = "%{patch_daal_kernel_defines}",
)

toolchain(
    name = "extra_tools_lnx",
    exec_compatible_with = [
        "@platforms//cpu:x86_64",
        "@platforms//os:linux",
    ],
    target_compatible_with = [
        "@platforms//cpu:x86_64",
        "@platforms//os:linux",
    ],
    toolchain = ":extra_tools",
    toolchain_type = "@onedal//dev/bazel/toolchains/extra",
)
