load("@onedal//dev/bazel:utils.bzl", "unique")

def _aggregate_deps(ctx):
    libs_to_link = []
    objects_to_link = []
    pic_objects_to_link = []
    link_flags = []
    for dep in ctx.attr.deps:
        local_link_ctx = dep[CcInfo].linking_context
        local_libs_to_link = local_link_ctx.libraries_to_link
        for lib_to_link in local_libs_to_link.to_list():
            if lib_to_link.objects:
                objects_to_link += lib_to_link.objects
            elif lib_to_link.pic_objects:
                pic_objects_to_link += lib_to_link.pic_objects
            else:
                # print(lib_to_link)
                libs_to_link.append(lib_to_link)
        link_flags += local_link_ctx.user_link_flags

    libraries_to_link = cc_common.create_linking_context(
        libraries_to_link = libs_to_link,
    ).libraries_to_link.to_list()
    if len(libraries_to_link) > 0:
        print(libraries_to_link[0])

    return struct(
        compilation_outputs = cc_common.create_compilation_outputs(
            objects = depset(objects_to_link),
            pic_objects = depset(pic_objects_to_link),
        ),
        linking_context = cc_common.create_linking_context(
            libraries_to_link = libs_to_link,
        ),
        user_link_flags = unique(link_flags),
    )


def _cc_shared_lib_impl(ctx):
    toolchain = ctx.toolchains["@bazel_tools//tools/cpp:toolchain_type"]
    feature_config = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = toolchain,
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )
    aggregated_deps = _aggregate_deps(ctx)
    # print(aggregated_deps)

    link_out = cc_common.link(
        name = ctx.label.name,
        actions = ctx.actions,
        cc_toolchain = toolchain,
        feature_configuration = feature_config,
        compilation_outputs = aggregated_deps.compilation_outputs,
        output_type	= "dynamic_library",
        linking_contexts = [ aggregated_deps.linking_context ],
        user_link_flags = aggregated_deps.user_link_flags,
    )

    dynamic_library_file = (link_out.library_to_link.resolved_symlink_dynamic_library or
                            link_out.library_to_link.dynamic_library)
    iface_library_file = (link_out.library_to_link.resolved_symlink_interface_library or
                          link_out.library_to_link.interface_library)
    library_files = [ dynamic_library_file ]
    if iface_library_file:
        library_files.append(iface_library_file)

    return [DefaultInfo(files = depset(library_files))]


cc_shared_lib = rule(
    implementation = _cc_shared_lib_impl,
    attrs = {
        "deps": attr.label_list(),
    },
    toolchains = ["@bazel_tools//tools/cpp:toolchain_type"],
    fragments = ["cpp"],
)
