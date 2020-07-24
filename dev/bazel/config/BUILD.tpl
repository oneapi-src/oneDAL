package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:cc.bzl", "cc_module")
load("@onedal//dev/bazel/config:config.bzl", "onedal_cpu_isa_extension_config")

onedal_cpu_isa_extension_config(
    name = "cpu",
    build_setting_default = "auto",
)

cc_module(
    name = "daal_version_data",
    hdrs = ["_daal_version_data.h"],
)
