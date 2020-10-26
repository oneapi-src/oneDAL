#===============================================================================
# Copyright 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

load("@onedal//dev/bazel:utils.bzl", "paths")

def _collect_java_infos(deps):
    java_infos = []
    for dep in deps:
        if JavaInfo in dep:
            java_infos.append(dep[JavaInfo])
    return java_infos

def _generate_jni_header_names(relative_to, srcs):
    jni_header_names = []
    for src in srcs:
        relative_path = paths.relativize(src.path, relative_to)
        path_components = relative_path.split('/')
        filename, extension = paths.split_extension(path_components[-1])
        if extension != ".java":
            fail("Expected `java` extension, but got `{}`".format(extension))
        path_components[-1] = filename + ".h"
        jni_header_name = "_".join(path_components)
        jni_header_names.append(jni_header_name)
    return jni_header_names

def _unpack_jni_headers(ctx, out_dir_name, native_headers, expected_jni_header_names):
    jdk_extra_tools = ctx.toolchains["@onedal//dev/bazel/toolchains/jdk:extra_tools"]
    expected_jni_header_files = []
    if expected_jni_header_names:
        for header_name in expected_jni_header_names:
            header_path = paths.join(out_dir_name, header_name)
            header_file = ctx.actions.declare_file(header_path)
            expected_jni_header_files.append(header_file)
        out_dir = expected_jni_header_files[0].dirname
        ctx.actions.run(
            inputs = [native_headers],
            outputs = expected_jni_header_files,
            executable = jdk_extra_tools.extract_jni_headers,
            arguments = [native_headers.path, out_dir] + expected_jni_header_names,
        )
    return expected_jni_header_files

def _java_jni_headers_impl(ctx):
    expected_jni_header_names = _generate_jni_header_names(
        ctx.attr.root_java_package_dir, ctx.files.srcs)
    generated_jni_headers = []
    for java_info in _collect_java_infos(ctx.attr.deps):
        out_dir_name = ctx.attr.out_dir_name
        native_headers = java_info.outputs.native_headers
        generated_jni_headers += _unpack_jni_headers(
            ctx, out_dir_name, native_headers, expected_jni_header_names)
    return [DefaultInfo(files=depset(generated_jni_headers))]


    # for src in ctx.files.srcs:
    #     print(src.path)

        # print(java_info.compilation_info.compilation_classpath)
        # print(java_info.compilation_info.runtime_classpath)


    # java_info = _collect_java_info(ctx.attr.deps)
    # print(java_info.outputs.jars)
    # native_headers = java_info.outputs.native_headers
    # return [ DefaultInfo(files=depset([native_headers])) ]

java_jni_headers = rule(
    implementation = _java_jni_headers_impl,
    attrs = {
        "srcs": attr.label_list(allow_files=True, mandatory=True),
        "deps": attr.label_list(mandatory=True),
        "out_dir_name": attr.string(mandatory=True),
        "root_java_package_dir": attr.string(default="java"),
    },
    toolchains = ["@onedal//dev/bazel/toolchains/jdk:extra_tools"],
)
