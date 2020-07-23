load("@bazel_tools//tools/cpp:toolchain_utils.bzl", "find_cpp_toolchain")
load("@onedal//dev/bazel/config:def.bzl", "CpuVectorInstructionsProvider")

def _categorize_sources(source_files, cpu_files_supported = True,
                                      fpt_files_supported = True):
    fpt_cpu_files_supported = cpu_files_supported and fpt_files_supported
    normal_files = []
    cpu_files = []
    fpt_files = []
    fpt_cpu_files = []
    for file in source_files:
        filename = file.basename
        if fpt_cpu_files_supported and ("_fpt_cpu" in filename):
            fpt_cpu_files.append(file)
        elif cpu_files_supported and ("_cpu" in filename):
            cpu_files.append(file)
        elif fpt_files_supported and ("_fpt" in filename):
            fpt_files.append(file)
        else:
            normal_files.append(file)
    return struct(
        normal_files = normal_files,
        cpu_files = cpu_files,
        fpt_files = fpt_files,
        fpt_cpu_files = fpt_cpu_files,
    )


def _compile(name, my_ctx, srcs, feature_config, local_defines=[]):
    return cc_common.compile(
        name = name,
        srcs = srcs,
        actions = my_ctx.ctx.actions,
        cc_toolchain = my_ctx.toolchain,
        local_defines = my_ctx.ctx.attr.local_defines + local_defines,
        user_compile_flags = my_ctx.ctx.attr.copts,
        compilation_contexts = my_ctx.dep_compilation_ctxs,
        feature_configuration = feature_config,
    )


def _compile_multidef(name, my_ctx):
    sources_by_category = _categorize_sources(
        source_files = my_ctx.ctx.files.srcs,
        cpu_files_supported = len(my_ctx.ctx.attr.cpus) > 0,
        fpt_files_supported = len(my_ctx.ctx.attr.fpts) > 0,
    )
    compilation_ctxs = []
    compilation_outs = []

    # Compile normal files
    cctx, cout = _compile(name, my_ctx, sources_by_category.normal_files, my_ctx.feature_config)
    compilation_ctxs.append(cctx)
    compilation_outs.append(cout)

    cpu_defines = my_ctx.ctx.attr.cpus
    enabled_cpus = my_ctx.ctx.attr.enabled_cpus[CpuVectorInstructionsProvider].isa_extensions
    cpu_defines_filtered = { k: cpu_defines[k] for k in enabled_cpus }

    cpu_feature_configs = {}
    for cpu in enabled_cpus:
        cpu_feature_configs[cpu] = cc_common.configure_features(
            ctx = my_ctx.ctx,
            cc_toolchain = my_ctx.toolchain,
            requested_features = my_ctx.ctx.features + [ "{}_flags".format(cpu) ],
            unsupported_features = my_ctx.ctx.disabled_features,
        )

    # Compile CPU files
    for cpu, defines in cpu_defines_filtered.items():
        cctx, cout = _compile(name + "_" + cpu, my_ctx, sources_by_category.cpu_files,
                              cpu_feature_configs[cpu], local_defines=defines)
        compilation_ctxs.append(cctx)
        compilation_outs.append(cout)

    # Compile FPT files
    for fpt, defines in my_ctx.ctx.attr.fpts.items():
        cctx, cout = _compile(name + "_" + fpt, my_ctx, sources_by_category.fpt_files,
                              cpu_feature_configs[cpu], local_defines=defines)
        compilation_ctxs.append(cctx)
        compilation_outs.append(cout)

    # Compile FPT-CPU files
    for cpu, cpu_defines in cpu_defines_filtered.items():
        for fpt, fpt_defines in my_ctx.ctx.attr.fpts.items():
            cctx, cout = _compile(name + "_" + fpt + "_" + cpu, my_ctx,
                                  sources_by_category.fpt_cpu_files,
                                  cpu_feature_configs[cpu],
                                  local_defines=cpu_defines + fpt_defines)
            compilation_ctxs.append(cctx)
            compilation_outs.append(cout)
    return compilation_ctxs, compilation_outs


def _create_cc_info_from_multiple_compilation_contexts(compilation_ctxs, linking_ctx):
    cc_infos = []
    for cctx in compilation_ctxs:
        local_cc_info = CcInfo(
            compilation_context = cctx,
            linking_context = linking_ctx
        )
        cc_infos.append(local_cc_info)
    return cc_common.merge_cc_infos(cc_infos=cc_infos)


def _create_my_ctx(ctx):
    toolchain = ctx.toolchains["@bazel_tools//tools/cpp:toolchain_type"]
    feature_config = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = toolchain,
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )
    dep_compilation_ctxs = []
    dep_linking_ctxs = []
    for dep in ctx.attr.deps:
        dep_compilation_ctxs.append(dep[CcInfo].compilation_context)
        dep_linking_ctxs.append(dep[CcInfo].linking_context)
    return struct(
        ctx = ctx,
        toolchain = toolchain,
        feature_config = feature_config,
        dep_compilation_ctxs = dep_compilation_ctxs,
        dep_linking_ctxs = dep_linking_ctxs,
    )


def _cc_multidef_lib_impl(ctx):
    my_ctx = _create_my_ctx(ctx)
    compilation_ctxs, compilation_outs = _compile_multidef(ctx.label.name, my_ctx)
    merged_compilation_outs = cc_common.merge_compilation_outputs(
        compilation_outputs = compilation_outs
    )
    linking_ctx, linking_out = cc_common.create_linking_context_from_compilation_outputs(
        name = ctx.label.name,
        actions = ctx.actions,
        cc_toolchain = my_ctx.toolchain,
        feature_configuration = my_ctx.feature_config,
        compilation_outputs = merged_compilation_outs,
        linking_contexts = my_ctx.dep_linking_ctxs,
    )
    files_to_build = (merged_compilation_outs.pic_objects +
                      merged_compilation_outs.objects)
    cc_info = _create_cc_info_from_multiple_compilation_contexts(
        compilation_ctxs,
        linking_ctx
    )
    default_info = DefaultInfo(
        files = depset(files_to_build)
    )
    return [default_info, cc_info]


cc_multidef_lib = rule(
    implementation = _cc_multidef_lib_impl,
    attrs = {
        "srcs": attr.label_list(allow_files=True, mandatory=True),
        "deps": attr.label_list(),
        "cpus": attr.string_list_dict(),
        "fpts": attr.string_list_dict(),
        "copts": attr.string_list(),
        "local_defines": attr.string_list(),
        "enabled_cpus": attr.label(
            default = "@config//:cpu",
        ),
        "_cc_toolchain": attr.label(
            default = Label("@onedal_cc_toolchain//:cc_toolchain")
        ),
    },
    toolchains = ["@bazel_tools//tools/cpp:toolchain_type"],
    fragments = ["cpp"],
)
