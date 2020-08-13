load("@onedal//dev/bazel/toolchains:common.bzl", "detect_os", "detect_compiler")
load("@onedal//dev/bazel/toolchains:extra_toolchain_lnx.bzl",
    "configure_extra_toolchain_lnx")

ExtraToolchainInfo = provider(
    fields = [
        # TODO
        "placeholder",
    ],
)

def _extra_toolchain_impl(ctx):
    toolchain_info = platform_common.ToolchainInfo(
        extra_toolchain_info = ExtraToolchainInfo(
            # TODO
            placeholder = "",
        ),
    )
    return [toolchain_info]

extra_toolchain = rule(
    implementation = _extra_toolchain_impl,
    attrs = {
        # TODO
    },
)

def _onedal_extra_toolchain_impl(repo_ctx):
    os_id = detect_os(repo_ctx)
    compiler_id = detect_compiler(repo_ctx, os_id)
    configure_extra_toolchain_os = {
        "lnx": configure_extra_toolchain_lnx,
    }[os_id]
    configure_extra_toolchain_os(repo_ctx, compiler_id)

onedal_extra_toolchain = repository_rule(
    implementation = _onedal_extra_toolchain_impl,
)

def declare_onedal_extra_toolchain(name):
    onedal_extra_toolchain(name = name)
    native.register_toolchains("@{}//:all".format(name))
