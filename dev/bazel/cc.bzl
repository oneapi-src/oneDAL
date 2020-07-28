load("@onedal//dev/bazel:utils.bzl",
    "utils",
    "paths",
    "sets",
)
load("@onedal//dev/bazel/config:config.bzl",
    "CpuVectorInstructionsProvider"
)

ModuleInfo = provider(
    fields=[
        "tagged_linking_contexts",
        "compilation_context",
    ]
)

def _collect_compilation_contexts(deps):
    dep_compilation_contexts = []
    for dep in deps:
        for Info in [CcInfo, ModuleInfo]:
            if Info in dep:
                dep_compilation_contexts.append(dep[Info].compilation_context)
    return dep_compilation_contexts

def _collect_tagged_linking_contexts(deps):
    dep_tagged_linking_contexts = []
    for dep in deps:
        if ModuleInfo in dep:
            dep_tagged_linking_contexts += dep[ModuleInfo].tagged_linking_contexts
        if CcInfo in dep:
            linking_context = dep[CcInfo].linking_context
            dep_tagged_linking_contexts.append((None, linking_context))
    return dep_tagged_linking_contexts

def _filter_tagged_linking_contexts(tagged_linking_contexts, tags):
    linking_contexts = []
    tag_set = sets.make(tags)
    for tag, linking_context in tagged_linking_contexts:
        if (not tag) or (not tags) or sets.contains(tag_set, tag):
            linking_contexts.append(linking_context)
    return linking_contexts

def _merge_compilation_contexts(compilation_contexts):
    cc_infos = [CcInfo(compilation_context=x) for x in compilation_contexts]
    return cc_common.merge_cc_infos(
        direct_cc_infos = cc_infos
    ).compilation_context

def _merge_linking_contexts(linking_contexts):
    link_flags = []
    libs_to_link = []
    objects_to_link = []
    pic_objects_to_link = []
    for linking_context in linking_contexts:
        for linker_input in linking_context.linker_inputs.to_list():
            for lib_to_link in linker_input.libraries:
                if lib_to_link.objects:
                    objects_to_link += lib_to_link.objects
                elif lib_to_link.pic_objects:
                    pic_objects_to_link += lib_to_link.pic_objects
                else:
                    libs_to_link.append(lib_to_link)
            link_flags += linker_input.user_link_flags
    return struct(
        objects = objects_to_link,
        pic_objects = pic_objects_to_link,
        libraries_to_link = libs_to_link,
        user_link_flags = utils.unique(link_flags),
    )

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

def _compile(name, ctx, toolchain, feature_config,
             dep_compilation_contexts, srcs=[], local_defines=[]):
    inc_dir = paths.dirname(ctx.build_file_path) + "/"
    gen_dir = ctx.genfiles_dir.path + "/" + inc_dir
    return cc_common.compile(
        name = name,
        srcs = srcs,
        actions = ctx.actions,
        public_hdrs = ctx.files.hdrs,
        private_hdrs = ctx.files.private_hdrs,
        cc_toolchain = toolchain,
        defines = ctx.attr.defines,
        local_defines = ctx.attr.local_defines + local_defines,
        user_compile_flags = ctx.attr.copts,
        includes = (utils.add_prefix(inc_dir, ctx.attr.includes) +
                    utils.add_prefix(gen_dir, ctx.attr.includes)),
        quote_includes = (utils.add_prefix(inc_dir, ctx.attr.quote_includes) +
                          utils.add_prefix(gen_dir, ctx.attr.quote_includes)),
        system_includes = (utils.add_prefix(inc_dir, ctx.attr.system_includes) +
                           utils.add_prefix(gen_dir, ctx.attr.system_includes)),
        compilation_contexts = dep_compilation_contexts,
        feature_configuration = feature_config,
        disallow_nopic_outputs = True,
    )

def _compile_all(name, ctx, toolchain, feature_config, dep_compilation_contexts):
    fpts = ctx.attr._fpts
    cpus = ctx.attr._cpus[CpuVectorInstructionsProvider].isa_extensions[:]
    if ctx.attr.disable_mic and "avx512_mic" in cpus:
        cpus.remove("avx512_mic")

    compilation_contexts = []
    compilation_outputs = []

    sources_by_category = _categorize_sources(
        source_files = ctx.files.srcs,
    )

    cpu_defines = {}
    cpu_feature_configs = {}
    if sources_by_category.cpu_files or sources_by_category.fpt_cpu_files:
        for cpu in cpus:
            cpu_defines[cpu] = ctx.attr.cpu_defines.get(cpu, [])
            cpu_feature_configs[cpu] = cc_common.configure_features(
                ctx = ctx,
                cc_toolchain = toolchain,
                requested_features = ctx.features + [ "{}_flags".format(cpu) ],
                unsupported_features = ctx.disabled_features,
            )

    fpt_defines = {}
    if sources_by_category.fpt_files or sources_by_category.fpt_cpu_files:
        for fpt in fpts:
            fpt_defines[fpt] = ctx.attr.fpt_defines.get(fpt, [])

    # Compile normal files
    compilation_context, compulation_output = _compile(
        name, ctx, toolchain, feature_config,
        dep_compilation_contexts,
        srcs = sources_by_category.normal_files
    )
    compilation_contexts.append(compilation_context)
    compilation_outputs.append(compulation_output)

    # Compile FPT files
    if sources_by_category.fpt_files:
        for fpt in fpts:
            compilation_context, compulation_output = _compile(
                name + "_" + fpt, ctx, toolchain, feature_config,
                dep_compilation_contexts,
                srcs = sources_by_category.fpt_files,
                local_defines = fpt_defines[fpt]
            )
            compilation_contexts.append(compilation_context)
            compilation_outputs.append(compulation_output)

    # Compile CPU files
    if sources_by_category.cpu_files:
        for cpu in cpus:
            compilation_context, compulation_output = _compile(
                name + "_" + cpu, ctx, toolchain, cpu_feature_configs[cpu],
                dep_compilation_contexts,
                srcs = sources_by_category.cpu_files,
                local_defines = cpu_defines[cpu]
            )
            compilation_contexts.append(compilation_context)
            compilation_outputs.append(compulation_output)

    # Compile FPT-CPU files
    if sources_by_category.fpt_cpu_files:
        for cpu in cpus:
            for fpt in fpts:
                compilation_context, compulation_output = _compile(
                    name + "_" + fpt + "_" + cpu, ctx, toolchain, cpu_feature_configs[cpu],
                    dep_compilation_contexts,
                    srcs = sources_by_category.fpt_cpu_files,
                    local_defines = cpu_defines[cpu] + fpt_defines[fpt]
                )
                compilation_contexts.append(compilation_context)
                compilation_outputs.append(compulation_output)

    compilation_context = _merge_compilation_contexts(compilation_contexts)
    compilation_output = cc_common.merge_compilation_outputs(
        compilation_outputs = compilation_outputs
    )
    return compilation_context, compilation_output


def _cc_module_impl(ctx):
    toolchain = ctx.toolchains["@bazel_tools//tools/cpp:toolchain_type"]
    feature_config = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = toolchain,
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )
    dep_compilation_contexts = _collect_compilation_contexts(ctx.attr.deps)
    compilation_context, compilation_outputs = _compile_all(
        name = ctx.label.name,
        ctx = ctx,
        toolchain = toolchain,
        feature_config = feature_config,
        dep_compilation_contexts = dep_compilation_contexts,
    )
    linking_context, linking_out = cc_common.create_linking_context_from_compilation_outputs(
        name = ctx.label.name,
        actions = ctx.actions,
        cc_toolchain = toolchain,
        feature_configuration = feature_config,
        compilation_outputs = compilation_outputs,
    )
    tagged_linking_contexts = _collect_tagged_linking_contexts(ctx.attr.deps)
    tagged_linking_contexts.append((ctx.attr.lib_tag, linking_context))
    files_to_build = (compilation_outputs.pic_objects +
                      compilation_outputs.objects)
    # default_info = DefaultInfo(
    #     files = depset(files_to_build)
    # )
    module_info = ModuleInfo(
        compilation_context = compilation_context,
        tagged_linking_contexts = tagged_linking_contexts,
    )
    # return [default_info, module_info]
    return [module_info]


_cc_module = rule(
    implementation = _cc_module_impl,
    attrs = {
        "lib_tag": attr.string(),
        "srcs": attr.label_list(allow_files=True),
        "hdrs": attr.label_list(allow_files=True),
        "private_hdrs": attr.label_list(allow_files=True),
        "deps": attr.label_list(),
        "copts": attr.string_list(),
        "defines": attr.string_list(),
        "local_defines": attr.string_list(),
        "cpu_defines": attr.string_list_dict(),
        "fpt_defines": attr.string_list_dict(),
        "includes": attr.string_list(),
        "quote_includes": attr.string_list(),
        "system_includes": attr.string_list(),
        "disable_mic": attr.bool(default=False),
        "_cpus": attr.label(
            default = "@config//:cpu",
        ),
        "_fpts": attr.string_list(default = ["f32", "f64"])
    },
    toolchains = ["@bazel_tools//tools/cpp:toolchain_type"],
    fragments = ["cpp"],
)

def cc_module(name, hdrs=[], deps=[], **kwargs):
    # Workaround restriction on possible extensions for cc_common.compile:
    # > The list of possible extensions for 'public_hdrs' is:
    #   .h,.hh,.hpp,.ipp,.hxx,.h++,.inc,.inl,.tlh,.tli,.H,.tcc
    native.cc_library(
        name = "__{}_headers__".format(name),
        hdrs = hdrs,
    )
    _cc_module(
        name = name,
        deps = [
            ":__{}_headers__".format(name),
        ] + deps,
        **kwargs,
    )

def _cc_static_lib_impl(ctx):
    toolchain = ctx.toolchains["@bazel_tools//tools/cpp:toolchain_type"]
    feature_config = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = toolchain,
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )
    tagged_linking_contexts = _collect_tagged_linking_contexts(ctx.attr.deps)
    linking_contexts = _filter_tagged_linking_contexts(tagged_linking_contexts, ctx.attr.lib_tags)
    merged_linking_context = _merge_linking_contexts(linking_contexts)
    object_files = depset(merged_linking_context.objects +
                          merged_linking_context.pic_objects)
    compilation_outputs = cc_common.create_compilation_outputs(
        objects = object_files,
        pic_objects = object_files,
    )
    linking_context, linking_outputs = cc_common.create_linking_context_from_compilation_outputs(
        name = ctx.attr.lib_name,
        actions = ctx.actions,
        cc_toolchain = toolchain,
        feature_configuration = feature_config,
        compilation_outputs = compilation_outputs,
        linking_contexts = linking_contexts,
        disallow_dynamic_library = True,
    )
    if not linking_outputs.library_to_link:
        return utils.warn("'{}' static library does not contain any " +
                          "object file".format(ctx.attr.lib_name))
    static_lib = (linking_outputs.library_to_link.static_library or
                  linking_outputs.library_to_link.pic_static_library)
    default_info = DefaultInfo(
        files = depset([ static_lib ]),
    )
    cc_info = CcInfo(
        compilation_context = None,
        linking_context = linking_context,
    )
    return [default_info, cc_info]

cc_static_lib = rule(
    implementation = _cc_static_lib_impl,
    attrs = {
        "lib_name": attr.string(),
        "lib_tags": attr.string_list(),
        "deps": attr.label_list(mandatory=True),
    },
    toolchains = ["@bazel_tools//tools/cpp:toolchain_type"],
    fragments = ["cpp"],
)

def _cc_executable_impl(ctx):
    toolchain = ctx.toolchains["@bazel_tools//tools/cpp:toolchain_type"]
    feature_config = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = toolchain,
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )
    tagged_linking_contexts = _collect_tagged_linking_contexts(ctx.attr.deps)
    linking_contexts = _filter_tagged_linking_contexts(tagged_linking_contexts, ctx.attr.lib_tags)
    merged_linking_context = _merge_linking_contexts(linking_contexts)
    object_files = depset(merged_linking_context.objects +
                          merged_linking_context.pic_objects)
    compilation_outputs = cc_common.create_compilation_outputs(
        objects = object_files,
        pic_objects = object_files,
    )
    linker_input = cc_common.create_linker_input(
        owner = ctx.label,
        libraries = depset(merged_linking_context.libraries_to_link),
        user_link_flags = depset(merged_linking_context.user_link_flags),
    )
    linking_outputs = cc_common.link(
        name = ctx.label.name,
        actions = ctx.actions,
        cc_toolchain = toolchain,
        feature_configuration = feature_config,
        # TODO: Pass compilations outputs via linking contexts
        #       Individual linking context for each library tag
        compilation_outputs = compilation_outputs,
        linking_contexts = [
            cc_common.create_linking_context(
                linker_inputs = depset([linker_input]),
            )
        ],
    )
    if not linking_outputs.executable:
        return utils.warn("'{}' executable does not contain any " +
                          "object file".format(ctx.label.name))
    default_info = DefaultInfo(
        files = depset([ linking_outputs.executable ]),
        executable = linking_outputs.executable,
    )
    return [default_info]

cc_executable = rule(
    implementation = _cc_executable_impl,
    attrs = {
        "lib_tags": attr.string_list(),
        "deps": attr.label_list(mandatory=True),
    },
    toolchains = ["@bazel_tools//tools/cpp:toolchain_type"],
    fragments = ["cpp"],
    executable = True,
)
