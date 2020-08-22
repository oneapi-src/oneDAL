load("@onedal//dev/bazel:utils.bzl",
    "utils",
    "paths",
    "sets",
)

_ModuleInfo = provider(
    fields=[
        "compilation_context",
        "tagged_linking_contexts",
    ]
)

def _collect_compilation_contexts(deps):
    dep_compilation_contexts = []
    for dep in deps:
        for Info in [CcInfo, _ModuleInfo]:
            if Info in dep:
                dep_compilation_contexts.append(dep[Info].compilation_context)
    return dep_compilation_contexts

def _merge_compilation_contexts(compilation_contexts):
    cc_infos = [CcInfo(compilation_context=x) for x in compilation_contexts]
    return cc_common.merge_cc_infos(
        direct_cc_infos = cc_infos
    ).compilation_context

def _collect_and_merge_compilation_contexts(deps):
    compilation_contexts = _collect_compilation_contexts(deps)
    compilation_contexts = _merge_compilation_contexts(compilation_contexts)
    return compilation_contexts

def _create_tagged_linking_context(tag, linking_context):
    return struct(
        tag = tag,
        linking_context = linking_context,
    )

def _collect_tagged_linking_contexts(deps):
    dep_tagged_linking_contexts = []
    for dep in deps:
        if _ModuleInfo in dep:
            dep_tagged_linking_contexts += dep[_ModuleInfo].tagged_linking_contexts
        if CcInfo in dep:
            linking_context = dep[CcInfo].linking_context
            dep_tagged_linking_contexts.append(_create_tagged_linking_context(
                tag = None,
                linking_context = linking_context,
            ))
    return dep_tagged_linking_contexts

def _filter_tagged_linking_contexts(tagged_linking_contexts, tags):
    linking_contexts = []
    tag_set = sets.make(tags)
    for tagged_linking_context in tagged_linking_contexts:
        tag = tagged_linking_context.tag
        linking_context = tagged_linking_context.linking_context
        if (not tag) or (not tags) or sets.contains(tag_set, tag):
            linking_contexts.append(linking_context)
    return linking_contexts

def _unpack_linking_contexts(linking_contexts):
    link_flags = []
    objects = []
    pic_objects = []
    dynamic_libs = []
    static_libs = []
    dynamic_libs_to_link = []
    static_libs_to_link = []
    for linking_context in linking_contexts:
        for linker_input in linking_context.linker_inputs.to_list():
            for lib_to_link in linker_input.libraries:
                if lib_to_link.objects:
                    objects += lib_to_link.objects
                elif lib_to_link.pic_objects:
                    pic_objects += lib_to_link.pic_objects
                elif lib_to_link.dynamic_library:
                    dynamic_libs_to_link.append(lib_to_link)
                    dynamic_libs.append(lib_to_link.dynamic_library)
                elif lib_to_link.static_library or lib_to_link.pic_static_library:
                    static_libs_to_link.append(lib_to_link)
                    static_libs.append(lib_to_link.static_library or
                                       lib_to_link.pic_static_library)
            link_flags += linker_input.user_link_flags
    if objects:
        fail("Non-PIC object files found, oneDAL assumes " +
             "all object files are compiled as PIC")
    return struct(
        objects = pic_objects,
        dynamic_libraries = dynamic_libs,
        dynamic_libraries_to_link = dynamic_libs_to_link,
        static_libraries = static_libs,
        static_libraries_to_link = static_libs_to_link,
        user_link_flags = utils.unique(link_flags),
    )

common = struct(
    ModuleInfo = _ModuleInfo,
    collect_compilation_contexts = _collect_compilation_contexts,
    merge_compilation_contexts = _merge_compilation_contexts,
    collect_and_merge_compilation_contexts = _collect_and_merge_compilation_contexts,
    create_tagged_linking_context = _create_tagged_linking_context,
    collect_tagged_linking_contexts = _collect_tagged_linking_contexts,
    filter_tagged_linking_contexts = _filter_tagged_linking_contexts,
    unpack_linking_contexts = _unpack_linking_contexts,
    filter_dynamic_libraries_to_link = _filter_dynamic_libraries_to_link,
    filter_static_libraries_to_link = _filter_static_libraries_to_link,
)
