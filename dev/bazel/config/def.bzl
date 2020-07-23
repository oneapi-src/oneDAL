load("@onedal//dev/bazel:utils.bzl", "unique")

CpuVectorInstructionsProvider = provider(
    fields = ["isa_extensions"]
)

_ISA_EXTENSIONS = ["sse2", "ssse3", "sse42", "avx", "avx2", "avx512", "avx512_mic"]

def _onedal_cpu_isa_extension_config_impl(ctx):
    # TODO: Transform ctx.build_setting_value -> ISA_EXTENSION_VALUE
    if ctx.build_setting_value == "auto":
        isa_extensions = _ISA_EXTENSIONS
    else:
        isa_extensions = ctx.build_setting_value.split(" ")
        isa_extensions = [x.strip() for x in isa_extensions]
        isa_extensions = unique(["sse2"] + isa_extensions)
    return CpuVectorInstructionsProvider(
        isa_extensions = isa_extensions
    )

onedal_cpu_isa_extension_config = rule(
    implementation = _onedal_cpu_isa_extension_config_impl,
    build_setting = config.string(flag = True)
)

def _declare_onedal_config_impl(repo_ctx):
    repo_ctx.template(
        "BUILD",
        Label("@onedal//dev/bazel/config:BUILD.tpl"),
    )

declare_onedal_config = repository_rule(
    implementation = _declare_onedal_config_impl,
)
