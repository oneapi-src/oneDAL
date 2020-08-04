load("@onedal//dev/bazel:utils.bzl", "utils", "sets")

CpuVectorInstructionsProvider = provider(
    fields = ["isa_extensions"]
)

_ISA_EXTENSIONS = ["sse2", "ssse3", "sse42", "avx", "avx2", "avx512", "avx512_mic"]
_ISA_EXTENSIONS_MODERN = ["sse2", "avx", "avx2", "avx512"]
_ISA_EXTENSION_AUTO_DEFAULT = "avx2"

def _check_cpu_extensions(extensions):
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
        isa_extensions = [ ctx.attr.auto_cpu ]
    else:
        isa_extensions = ctx.build_setting_value.split(" ")
        isa_extensions = [x.strip() for x in isa_extensions]
    isa_extensions = utils.unique(["sse2"] + isa_extensions)
    _check_cpu_extensions(isa_extensions)
    return CpuVectorInstructionsProvider(
        isa_extensions = isa_extensions
    )

onedal_cpu_isa_extension_config = rule(
    implementation = _onedal_cpu_isa_extension_config_impl,
    build_setting = config.string(flag = True),
    attrs = {
        "auto_cpu": attr.string(mandatory=True),
    },
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
    repo_ctx.report_progress("Generate DAAL version header")
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

def _detect_cpu_extension(repo_ctx):
    cpudetect_src = repo_ctx.path(repo_ctx.attr._cpudetect_src)
    cpudetect_exe = repo_ctx.path("cpudetect")
    repo_ctx.report_progress("Compile cpu-detector")
    compile_result = repo_ctx.execute([
        "g++", "-pedantic", "-Wall", "-std=c++11",
        cpudetect_src, "-o{}".format(cpudetect_exe),
    ])
    if compile_result.return_code != 0:
        utils.warn("Cannot compile cpu-detector:\n" +
                   compile_result.stderr + "\n" +
                   "Use {} by default.".format(_ISA_EXTENSION_AUTO_DEFAULT))
        return _ISA_EXTENSION_AUTO_DEFAULT
    repo_ctx.report_progress("Run cpu-detector to determine default vector extension")
    cpudetect_result = repo_ctx.execute([cpudetect_exe])
    if cpudetect_result.return_code != 0:
        utils.warn("Cannot run cpu-detector:\n" +
                   cpudetect_result.stderr + "\n" +
                   "Use {} by default.".format(_ISA_EXTENSION_AUTO_DEFAULT))
        return _ISA_EXTENSION_AUTO_DEFAULT
    return cpudetect_result.stdout.strip()

def _declare_onedal_config_impl(repo_ctx):
    _generate_daal_version_data(repo_ctx)
    repo_ctx.template(
        "BUILD",
        Label("@onedal//dev/bazel/config:BUILD.tpl"),
        substitutions = {
            "%{auto_cpu}": _detect_cpu_extension(repo_ctx),
        },
    )

declare_onedal_config = repository_rule(
    implementation = _declare_onedal_config_impl,
    attrs = {
        "_cpudetect_src": attr.label(
            default = "@onedal//dev/bazel/config:cpudetect.cpp",
        ),
    },
)
