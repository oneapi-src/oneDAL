load("@onedal//dev/bazel:utils.bzl", "utils", "sets")

CpuInfo = provider(
    fields = ["isa_extensions"]
)

VersionInfo = provider(
    fields = [
        "major",
        "minor",
        "update",
        "build",
        "buildrev",
        "status",
    ]
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


def _cpu_info_impl(ctx):
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
    return CpuInfo(
        isa_extensions = isa_extensions
    )

cpu_info = rule(
    implementation = _cpu_info_impl,
    build_setting = config.string(flag = True),
    attrs = {
        "auto_cpu": attr.string(mandatory=True),
    },
)

def _version_info_impl(ctx):
    return [
        VersionInfo(
            major    = ctx.attr.major,
            minor    = ctx.attr.minor,
            update   = ctx.attr.update,
            build    = ctx.attr.build,
            buildrev = ctx.attr.buildrev,
            status   = ctx.attr.status,
        )
    ]

version_info = rule(
    implementation = _version_info_impl,
    attrs = {
        "major": attr.string(mandatory=True),
        "minor": attr.string(mandatory=True),
        "update": attr.string(mandatory=True),
        "build": attr.string(mandatory=True),
        "buildrev": attr.string(mandatory=True),
        "status": attr.string(mandatory=True),
    },
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
    repo_ctx.template(
        "BUILD",
        Label("@onedal//dev/bazel/config:BUILD.tpl"),
        substitutions = {
            "%{auto_cpu}": _detect_cpu_extension(repo_ctx),
            "%{version_major}":    "2021",
            "%{version_minor}":    "1",
            "%{version_update}":   "8",
            "%{version_build}":    utils.datestamp(repo_ctx),
            "%{version_buildrev}": "work",
            "%{version_status}":   "B",
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
