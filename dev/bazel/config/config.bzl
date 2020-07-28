load("@onedal//dev/bazel:utils.bzl", "utils", "sets")

CpuVectorInstructionsProvider = provider(
    fields = ["isa_extensions"]
)

_ISA_EXTENSIONS = ["sse2", "ssse3", "sse42", "avx", "avx2", "avx512", "avx512_mic"]
_ISA_EXTENSIONS_MODERN = ["sse2", "avx", "avx2", "avx512"]

def _check_extensions(extensions):
    allowed = sets.make(_ISA_EXTENSIONS)
    requested = sets.make(extensions)
    unsupported = sets.to_list(sets.difference(requested, allowed))
    if unsupported:
        fail("Unsupported CPU extensions: {}\n".format(unsupported) +
             "Allowed extensions: {}".format(_ISA_EXTENSIONS))

def _onedal_cpu_isa_extension_config_impl(ctx):
    if ctx.build_setting_value == "all":
        isa_extensions = _ISA_EXTENSIONS
    elif ctx.build_setting_value == "modern":
        isa_extensions = _ISA_EXTENSIONS_MODERN
    elif ctx.build_setting_value == "auto":
        # TODO: Determine CPU on build machine
        isa_extensions = _ISA_EXTENSIONS_MODERN
    else:
        isa_extensions = ctx.build_setting_value.split(" ")
        isa_extensions = [x.strip() for x in isa_extensions]
        isa_extensions = utils.unique(["sse2"] + isa_extensions)
    _check_extensions(isa_extensions)
    return CpuVectorInstructionsProvider(
        isa_extensions = isa_extensions
    )

onedal_cpu_isa_extension_config = rule(
    implementation = _onedal_cpu_isa_extension_config_impl,
    build_setting = config.string(flag = True)
)

def _version_info(repo_ctx):
    # TODO: Read version information from file
    return dict(
        major    = "2021",
        minor    = "1",
        update   = "8",
        build    = utils.datestamp(repo_ctx),
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
