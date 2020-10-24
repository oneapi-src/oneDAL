load("@bazel_tools//tools/jdk:toolchain_utils.bzl",
    "find_java_toolchain",
    "find_java_runtime_toolchain",
)

_implicit_deps = {
    "_java_toolchain": attr.label(
        default = Label("@bazel_tools//tools/jdk:current_java_toolchain"),
    ),
    "_host_javabase": attr.label(
        default = Label("@bazel_tools//tools/jdk:current_java_runtime"),
        cfg = "host",
    ),
    "_java_runtime": attr.label(
        default = Label("@bazel_tools//tools/jdk:current_java_runtime"),
    ),
}

def _java_rule_attrs(attrs):
    concat_attrs = {}
    concat_attrs.update(attrs)
    concat_attrs.update(_implicit_deps)
    return concat_attrs


def _java_jni_headers_impl(ctx):
    java_toolchain = find_java_toolchain(ctx, ctx.attr._java_toolchain)
    host_javabase = find_java_runtime_toolchain(ctx, ctx.attr._host_javabase)
    full_java_jar = ctx.actions.declare_file(ctx.label.name + ".jar")
    java_info = java_common.compile(
        ctx,
        java_toolchain = java_toolchain,
        # TODO: `host_javabase` will be deprecated in the next version
        host_javabase = host_javabase,
        source_files = ctx.files.srcs,
        output = full_java_jar,
    )
    native_headers = java_info.outputs.native_headers
    print(native_headers)
    return [ DefaultInfo(files=depset([native_headers])) ]

java_jni_headers = rule(
    implementation = _java_jni_headers_impl,
    attrs = _java_rule_attrs({
        "srcs": attr.label_list(allow_files=True, mandatory=True),
    }),
    # Toolchain resolution for Java is not implemented, issue to track this
    # https://github.com/bazelbuild/bazel/issues/4592
    # toolchains = ["@bazel_tools//tools/jdk:toolchain_type"],
    fragments = ["java"],
)
