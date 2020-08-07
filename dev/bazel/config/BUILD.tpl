package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel/config:config.bzl",
    "cpu_info",
    "version_info",
    "config_flag",
)

cpu_info(
    name = "cpu",
    auto_cpu = "%{auto_cpu}",
    build_setting_default = "auto",
)

version_info(
    name = "version",
    major = "%{version_major}",
    minor = "%{version_minor}",
    update = "%{version_update}",
    build = "%{version_build}",
    buildrev = "%{version_buildrev}",
    status = "%{version_status}",
)

config_flag(
    name = "test_link_mode",
    build_setting_default = "dev",
)

config_setting(
    name = "dev_test_link_mode",
    flag_values  = {
        ":test_link_mode": "dev",
    },
)

config_setting(
    name = "static_test_link_mode",
    flag_values  = {
        ":test_link_mode": "static",
    },
)

config_setting(
    name = "dynamic_test_link_mode",
    flag_values  = {
        ":test_link_mode": "dynamic",
    },
)

config_flag(
    name = "test_thread_mode",
    build_setting_default = "par",
)

config_setting(
    name = "par_test_thread_mode",
    flag_values  = {
        ":test_thread_mode": "par",
    },
)

config_setting(
    name = "seq_test_thread_mode",
    flag_values  = {
        ":test_thread_mode": "seq",
    },
)
