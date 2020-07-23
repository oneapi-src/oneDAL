package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel/config:def.bzl", "onedal_cpu_isa_extension_config")

onedal_cpu_isa_extension_config(
    name = "cpu",
    build_setting_default = "auto",
)
