load("@onedal//dev/bazel:cc.bzl",
    "cc_module",
    "cc_static_lib",
    "cc_executable",
)
load("@onedal//dev/bazel:utils.bzl",
    "sets",
)
load("@onedal//dev/bazel/config:config.bzl",
    "CpuInfo",
)

def dal_module(name, features=[], hdrs=[], srcs=[],
               host=True, dpc=False, auto=False, **kwargs):
    hdrs_cc = []
    srcs_cc = []
    hdrs_dpc = []
    srcs_dpc = []
    if auto:
        hpp_filt = ["**/*.hpp"]
        cpp_filt = ["**/*.cpp"]
        dpc_filt = ["**/*_dpc.cpp"]
        test_filt = ["**/*_test*"]
        hdrs_all = native.glob(hpp_filt, exclude=test_filt)
        hdrs_cc = hdrs_all
        hdrs_dpc = hdrs_all
        srcs_cc = native.glob(cpp_filt, exclude=test_filt + dpc_filt)
        srcs_dpc = native.glob(cpp_filt, exclude=test_filt)
    if host:
        _dal_module(
            name = name,
            features = features,
            hdrs = hdrs_cc + hdrs,
            srcs = srcs_cc + srcs,
            **kwargs,
        )
    if dpc:
        _dal_module(
            name = name + "_dpc",
            features = ["dpc++"] + features,
            hdrs = hdrs_dpc + hdrs,
            srcs = srcs_dpc + srcs,
            local_defines = [ "ONEAPI_DAL_DATA_PARALLEL" ],
            **kwargs,
        )

def dal_static_lib(name, lib_name, host=True, dpc=False, deps=[],
                   lib_tags=["dal"], external_deps=[], **kwargs):
    if host:
        cc_static_lib(
            name = name,
            lib_name = lib_name,
            lib_tags = lib_tags,
            deps = deps + external_deps,
            **kwargs
        )
    if dpc:
        cc_static_lib(
            name = name + "_dpc",
            lib_name = lib_name + "_dpc",
            lib_tags = lib_tags,
            deps = [ d + "_dpc" for d in deps ] + external_deps,
            **kwargs
        )

def dal_executable(name, lib_tags=[], **kwargs):
    _dal_module(
        name = name + "_module",
        **kwargs,
    )
    cc_executable(
        name = name,
        deps = [":{}_module".format(name)],
        lib_tags = lib_tags,
    )

def dal_test(name, deps=[], test_deps=[], **kwargs):
    cc_static_lib(
        name = name + "_static",
        lib_name = "onedal_" + name,
        deps = deps,
    )
    dal_executable(
        name = name,
        deps = select({
            "@config//:dev_test_link_mode": [
                ":" + name + "_static",
            ],
            "@config//:static_test_link_mode": [
                "@onedal//cpp/oneapi/dal:static",
                "@onedal//cpp/daal:core_static",
            ],
            "@config//:dynamic_test_link_mode": [],
        }) +
        select({
            "@config//:par_test_thread_mode": [
                "@onedal//cpp/daal:thread_static",
            ],
            "@config//:seq_test_thread_mode": [
                "@onedal//cpp/daal:sequential_static",
            ],
        }) + test_deps,
        **kwargs,
    )

def dal_algo_shortcuts(*algos):
    for algo in algos:
        _dal_module(
            name = algo,
            hdrs = [ "{}.hpp".format(algo) ],
            deps = [ "@onedal//cpp/oneapi/dal/algo/{0}:{0}".format(algo) ],
        )
        _dal_module(
            name = algo + "_dpc",
            hdrs = [ "{}.hpp".format(algo) ],
            deps = [ "@onedal//cpp/oneapi/dal/algo/{0}:{0}_dpc".format(algo) ],
        )

def _dal_module(name, lib_tag="dal", features=[], **kwargs):
    cc_module(
        name = name,
        lib_tag = lib_tag,
        features = [ "pedantic", "c++17" ] + features,
        disable_mic = True,
        **kwargs,
    )

def _dal_generate_cpu_dispatcher_impl(ctx):
    cpus = sets.make(ctx.attr._cpus[CpuInfo].isa_extensions)
    content = (
        "// DO NOT EDIT: file is auto-generated on build time\n" +
        "// DO NOT PUT THIS FILE TO SVC: file is auto-generated on build time\n" +
        "// CPU detection logic specified in dev/bazel/config.bzl file\n" +
        "\n" +
        ("#define ONEDAL_CPU_DISPATCH_SSSE3\n"      if sets.contains(cpus, "ssse3")      else "") +
        ("#define ONEDAL_CPU_DISPATCH_SSE42\n"      if sets.contains(cpus, "sse42")      else "") +
        ("#define ONEDAL_CPU_DISPATCH_AVX\n"        if sets.contains(cpus, "avx")        else "") +
        ("#define ONEDAL_CPU_DISPATCH_AVX2\n"       if sets.contains(cpus, "avx2")       else "") +
        ("#define ONEDAL_CPU_DISPATCH_AVX512\n"     if sets.contains(cpus, "avx512")     else "")
    )
    kernel_defines = ctx.actions.declare_file(ctx.attr.out)
    ctx.actions.write(kernel_defines, content)
    return [ DefaultInfo(files=depset([ kernel_defines ])) ]

dal_generate_cpu_dispatcher = rule(
    implementation = _dal_generate_cpu_dispatcher_impl,
    output_to_genfiles = True,
    attrs = {
        "out": attr.string(mandatory=True),
        "_cpus": attr.label(
            default = "@config//:cpu",
        ),
    },
)
