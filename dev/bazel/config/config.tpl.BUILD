package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel/config:config.bzl",
    "cpu_info",
    "version_info",
    "config_flag",
    "config_bool_flag",
    "dump_config_info",
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
    allowed_build_setting_values = [
        "dev",
        "static",
        "dynamic",
        "release_static",
        "release_dynamic",
    ],
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

config_setting(
    name = "release_static_test_link_mode",
    flag_values  = {
        ":test_link_mode": "release_static",
    },
)

config_setting(
    name = "release_dynamic_test_link_mode",
    flag_values  = {
        ":test_link_mode": "release_dynamic",
    },
)

config_flag(
    name = "test_thread_mode",
    build_setting_default = "par",
    allowed_build_setting_values = [
        "par",
        "seq",
    ],
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

config_bool_flag(
    name = "release_dpc",
    build_setting_default = False,
)

config_setting(
    name = "release_dpc_enabled",
    flag_values  = {
        ":release_dpc": "True",
    },
)

dump_config_info(
    name = "dump",
    cpu_info = ":cpu",
    version_info = ":version",
    flags = [
        ":test_link_mode",
        ":test_thread_mode",
    ],
)
