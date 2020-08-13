
def configure_extra_toolchain_lnx(repo_ctx, compiler_id):
    repo_ctx.template(
        "BUILD",
        Label("@onedal//dev/bazel/toolchains:extra_toolchian_lnx.tpl.BUILD"),
    )
