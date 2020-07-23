load("@bazel_skylib//lib:paths.bzl", "paths")
load("@onedal//dev/bazel:utils.bzl", "utils")

TransitiveCcInfo = provider(
    fields=[
        "tagged_linking_contexts",
        "compilation_context",
    ]
)

def _collect_compilation_contexts(deps):
    dep_compilation_contexts = []
    for dep in deps:
        for Info in [CcInfo, TransitiveCcInfo]:
            if Info in dep:
                dep_compilation_contexts.append(dep[Info].compilation_context)
    return dep_compilation_contexts

def _collect_tagged_linking_contexts(deps):
    dep_tagged_linking_contexts = []
    for dep in deps:
        if TransitiveCcInfo in dep:
            dep_tagged_linking_contexts += dep[TransitiveCcInfo].tagged_linking_contexts
        if CcInfo in dep:
            linking_context = dep[CcInfo].linking_context
            dep_tagged_linking_contexts.append((None, linking_context))
    return dep_tagged_linking_contexts

def _filter_tagged_linking_contexts(tagged_linking_contexts, tags):
    linking_contexts = []
    tag_set = utils.set(tags)
    for tag, linking_context in tagged_linking_contexts:
        if (not tag) or (tag in tag_set):
            linking_contexts.append(linking_context)
    return linking_contexts

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


def _cc_module_impl(ctx):
    toolchain = ctx.toolchains["@bazel_tools//tools/cpp:toolchain_type"]
    feature_config = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = toolchain,
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )
    inc_dir = paths.dirname(ctx.build_file_path) + "/"
    gen_dir = ctx.genfiles_dir.path + "/" + inc_dir
    compilation_context, compilation_out = cc_common.compile(
        name = ctx.label.name,
        srcs = ctx.files.srcs,
        actions = ctx.actions,
        public_hdrs = ctx.files.hdrs,
        cc_toolchain = toolchain,
        defines = ctx.attr.defines,
        local_defines = ctx.attr.local_defines,
        user_compile_flags = ctx.attr.copts,
        includes = (utils.add_prefix(inc_dir, ctx.attr.includes) +
                    utils.add_prefix(gen_dir, ctx.attr.includes)),
        quote_includes = (utils.add_prefix(inc_dir, ctx.attr.quote_includes) +
                          utils.add_prefix(gen_dir, ctx.attr.quote_includes)),
        system_includes = (utils.add_prefix(inc_dir, ctx.attr.system_includes) +
                           utils.add_prefix(gen_dir, ctx.attr.system_includes)),
        compilation_contexts = _collect_compilation_contexts(ctx.attr.deps),
        feature_configuration = feature_config,
        disallow_nopic_outputs = True,
    )
    linking_context, linking_out = cc_common.create_linking_context_from_compilation_outputs(
        name = ctx.label.name,
        actions = ctx.actions,
        cc_toolchain = toolchain,
        feature_configuration = feature_config,
        compilation_outputs = compilation_out,
    )
    tagged_linking_contexts = _collect_tagged_linking_contexts(ctx.attr.deps)
    tagged_linking_contexts.append((ctx.attr.lib_tag, linking_context))
    files_to_build = (compilation_out.pic_objects +
                      compilation_out.objects)
    default_info = DefaultInfo(
        files = depset(files_to_build)
    )
    cc_info = TransitiveCcInfo(
        compilation_context = compilation_context,
        tagged_linking_contexts = tagged_linking_contexts,
    )
    return [default_info, cc_info]


_cc_module = rule(
    implementation = _cc_module_impl,
    attrs = {
        "lib_tag": attr.string(),
        "srcs": attr.label_list(allow_files=True),
        "hdrs": attr.label_list(allow_files=True),
        "libs": attr.label_list(allow_files=True),
        "deps": attr.label_list(),
        "copts": attr.string_list(),
        "defines": attr.string_list(),
        "local_defines": attr.string_list(),
        "includes": attr.string_list(),
        "quote_includes": attr.string_list(),
        "system_includes": attr.string_list(),
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
        utils.warn("'{}' static library does not contain any object file".format(ctx.attr.lib_name))
        return
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
