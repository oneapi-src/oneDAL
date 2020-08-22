load("@onedal//dev/bazel:utils.bzl",
    "utils",
    "paths",
    "sets",
)
load("@onedal//dev/bazel/toolchains:action_names.bzl",
    "CPP_MERGE_STATIC_LIBRARIES"
)
load("@onedal//dev/bazel/cc:common.bzl",
    onedal_cc_common = "common",
)

def _merge_static_libs(filename, actions, cc_toolchain,
                       feature_configuration, static_libs):
    output_file = actions.declare_file(filename)
    merger_path = cc_common.get_tool_for_action(
        feature_configuration = feature_configuration,
        action_name = CPP_MERGE_STATIC_LIBRARIES,
    )
    merger_variables = cc_common.create_link_variables(
        feature_configuration = feature_configuration,
        cc_toolchain = cc_toolchain,
        is_using_linker = False,
    )
    command_line = cc_common.get_memory_inefficient_command_line(
        feature_configuration = feature_configuration,
        action_name = CPP_MERGE_STATIC_LIBRARIES,
        variables = merger_variables,
    )
    args = actions.args()
    args.add_all(command_line)
    args.add(output_file)
    args.add_all(static_libs)
    env = cc_common.get_environment_variables(
        feature_configuration = feature_configuration,
        action_name = CPP_MERGE_STATIC_LIBRARIES,
        variables = merger_variables,
    )
    actions.run(
        executable = merger_path,
        arguments = [args],
        env = env,
        inputs = depset(
            direct = static_libs,
            transitive = [
                cc_toolchain.all_files,
            ],
        ),
        outputs = [output_file],
    )
    return output_file

def _link_static(owner, name, actions, cc_toolchain,
                 feature_configuration, linking_contexts):
    unpacked_linking_context = onedal_cc_common.unpack_linking_contexts(linking_contexts)
    compilation_outputs = cc_common.create_compilation_outputs(
        objects = depset(unpacked_linking_context.objects),
        pic_objects = depset(unpacked_linking_context.objects),
    )
    _, linking_outputs = cc_common.create_linking_context_from_compilation_outputs(
        name = name + ("_no_deps" if unpacked_linking_context.static_libraries else ""),
        actions = actions,
        cc_toolchain = cc_toolchain,
        feature_configuration = feature_configuration,
        compilation_outputs = compilation_outputs,
        linking_contexts = linking_contexts,
        disallow_dynamic_library = True,
    )
    if not linking_outputs.library_to_link:
        return utils.warn("'{}' static library does not contain any " +
                          "object file".format(name))
    static_lib = (linking_outputs.library_to_link.static_library or
                  linking_outputs.library_to_link.pic_static_library)
    if unpack_linking_contexts.static_libraries:
        static_lib = _merge_static_libs(
            filename = utils.remove_substring(static_lib.basename, "_no_deps")
            actions = actions,
            cc_toolchain = cc_toolchain,
            feature_configuration = feature_configuration,
            static_libs = [ static_lib ] + unpack_linking_contexts.static_libraries,
        )
    static_lib_to_link = cc_common.create_library_to_link(
        actions = actions,
        cc_toolchain = cc_toolchain,
        feature_configuration = feature_configuration,
        static_library = static_lib,
        pic_static_library = static_lib,
    )
    linker_input = cc_common.create_linker_input(
        owner = owner,
        libraries = depset([static_lib_to_link] +
                           unpack_linking_contexts.dynamic_libraries_to_link),
        user_link_flags = depset(unpacked_linking_context.user_link_flags),
    )
    linking_context = cc_common.create_linking_context(
        linker_inputs = depset([ linker_input ]),
    )
    return linking_context, static_lib

def _link_executable(owner, name, actions, cc_toolchain,
                     feature_configuration, linking_contexts):
    toolchain = ctx.toolchains["@bazel_tools//tools/cpp:toolchain_type"]
    feature_config = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = toolchain,
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )
    tagged_linking_contexts = onedal_cc_common.collect_tagged_linking_contexts(ctx.attr.deps)
    linking_contexts = onedal_cc_common.filter_tagged_linking_contexts(
        tagged_linking_contexts, ctx.attr.lib_tags)
    merged_linking_context = onedal_cc_common.merge_linking_contexts(linking_contexts)
    if merged_linking_context.objects:
        fail("Non-PIC object files found, oneDAL assumes " +
             "all object files are compiled as PIC")
    object_files = depset(merged_linking_context.pic_objects)
    compilation_outputs = cc_common.create_compilation_outputs(
        objects = object_files,
        pic_objects = object_files,
    )
    linker_input = cc_common.create_linker_input(
        owner = ctx.label,
        libraries = depset(merged_linking_context.libraries_to_link),
        user_link_flags = depset(merged_linking_context.user_link_flags),
    )
    # TODO: Pass compilations outputs via linking contexts
    #       Individual linking context for each library tag
    linking_context = cc_common.create_linking_context(
        linker_inputs = depset([linker_input]),
    )
    linking_outputs = cc_common.link(
        name = ctx.label.name,
        actions = ctx.actions,
        cc_toolchain = toolchain,
        feature_configuration = feature_config,
        compilation_outputs = compilation_outputs,
        linking_contexts = [linking_context],
    )
    if not linking_outputs.executable:
        return utils.warn("'{}' executable does not contain any " +
                          "object file".format(ctx.label.name))
    return linking_outputs.executable

link = struct(
    link_static = _link_static,
    link_executable = _link_executable,
)
