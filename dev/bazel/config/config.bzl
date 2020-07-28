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

def _datestamp(repo_ctx):
    return repo_ctx.execute(["date", "+%Y%m%d"]).stdout.strip()

def _version_info(repo_ctx):
    # TODO: Read version information from file
    return dict(
        major    = "2021",
        minor    = "1",
        update   = "8",
        build    = _datestamp(repo_ctx),
        buildrev = "work",
        status   = "B",
    )

def _generate_daal_version_data(repo_ctx):
    content = (
        "// DO NOT EDIT: file is auto-generated on build time\n" +
        "// DO NOT PUT THIS FILE TO SVN: file is auto-generated on build time\n" +
        "// Product version is specified in src/makefile.ver file\n" +
        "#define MAJORVERSION {major}\n" +
        "#define MINORVERSION {minor}\n" +
        "#define UPDATEVERSION {update}\n" +
        "#define BUILD \"{build}\"\n" +
        "#define BUILD_REV \"{buildrev}\"\n" +
        "#define PRODUCT_STATUS '{status}'\n"
    )
    version_info = _version_info(repo_ctx)
    repo_ctx.file(
        "_daal_version_data.h",
        executable = False,
        content = content.format(**version_info),
    )

def _declare_onedal_config_impl(repo_ctx):
    _generate_daal_version_data(repo_ctx)
    repo_ctx.template(
        "BUILD",
        Label("@onedal//dev/bazel/config:BUILD.tpl"),
    )

declare_onedal_config = repository_rule(
    implementation = _declare_onedal_config_impl,
)
