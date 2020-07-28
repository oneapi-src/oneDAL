load("@onedal//dev/bazel:cc.bzl",
    "cc_module",
    "cc_static_lib",
    "cc_executable",
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

def dal_cpu_dispatcher(name, src, out):
    _patch_cpu_dispatcher(
        name = name + "_patch",
        src = src,
        out = out,
    )
    _dal_module(
        name = name,
        hdrs = [
            ":" + name + "_patch"
        ]
    )

def _dal_module(name, lib_tag="dal", features=[], **kwargs):
    cc_module(
        name = name,
        lib_tag = lib_tag,
        features = [ "pedantic", "c++17" ] + features,
        disable_mic = True,
        **kwargs,
    )

def _patch_cpu_dispatcher_impl(ctx):
    # TODO: Patch source file
    patched_file = ctx.actions.declare_file(ctx.attr.out)
    ctx.actions.run(
        outputs = [ patched_file ],
        inputs = [ ctx.file.src ],
        executable = "cp",
        arguments = [ ctx.file.src.path, patched_file.path ]
    )
    return [
        DefaultInfo(
            files = depset([ patched_file ])
        )
    ]

_patch_cpu_dispatcher = rule(
    implementation = _patch_cpu_dispatcher_impl,
    output_to_genfiles = True,
    attrs = {
        "src": attr.label(allow_single_file=True, mandatory=True),
        "out": attr.string(mandatory=True),
    },
)
