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

load("@onedal//dev/bazel:utils.bzl", "utils", "paths")
load("@onedal//dev/bazel:cc.bzl", "ModuleInfo")

def _match_file_name(file, entries):
    for entry in entries:
        if entry in file.path:
            return True
    return False

def _collect_headers(dep):
    headers = []
    if ModuleInfo in dep:
        headers += dep[ModuleInfo].compilation_context.headers.to_list()
    elif CcInfo in dep:
        headers += dep[CcInfo].compilation_context.headers.to_list()
    elif DefaultInfo in dep:
        headers += dep[DefaultInfo].files.to_list()
    return utils.unique_files(headers)

def _collect_default_files(deps):
    files = []
    for dep in deps:
        if DefaultInfo in dep:
            files += dep[DefaultInfo].files.to_list()
    return utils.unique_files(files)

def _copy(ctx, src_file, dst_path):
    # TODO: Use extra toolchain
    dst_file = ctx.actions.declare_file(dst_path)
    ctx.actions.run(
        executable = "cp",
        inputs = [ src_file ],
        outputs = [ dst_file ],
        use_default_shell_env = True,
        arguments = [ src_file.path, dst_file.path ],
    )
    return dst_file

def _try_relativize(path, start):
    if path.startswith(start):
        return paths.relativize(path, start)
    return path

def _copy_include(ctx, prefix):
    include_prefix = paths.join(prefix, "include")
    dst_files = []
    for include, prefix, skip_prefix in zip(ctx.attr.include, ctx.attr.include_prefix,
                                            ctx.attr.include_skip_prefix):
        headers = _collect_headers(include)
        for header in headers:
            if skip_prefix:
                dst_path = _try_relativize(header.path, skip_prefix)
            elif prefix:
                dst_path = paths.join(prefix, header.basename)
            dst_file = _copy(ctx, header, paths.join(include_prefix, dst_path))
            dst_files.append(dst_file)
    return dst_files

def _copy_lib(ctx, prefix):
    lib_prefix = paths.join(prefix, "lib", "intel64")
    libs = _collect_default_files(ctx.attr.lib)
    dst_files = []
    for lib in libs:
        dst_path = paths.join(lib_prefix, lib.basename)
        dst_file = _copy(ctx, lib, dst_path)
        dst_files.append(dst_file)
    return dst_files

def _copy_to_release_impl(ctx):
    extra_toolchain = ctx.toolchains["@onedal//dev/bazel/toolchains:extra"]
    prefix = ctx.attr.name + "/daal/latest"
    files = []
    files += _copy_include(ctx, prefix)
    files += _copy_lib(ctx, prefix)
    return [DefaultInfo(files=depset(files))]

_release = rule(
    implementation = _copy_to_release_impl,
    attrs = {
        "include": attr.label_list(allow_files=True),
        "include_prefix": attr.string_list(),
        "include_skip_prefix": attr.string_list(),
        "lib": attr.label_list(allow_files=True),
    },
    toolchains = [
        "@onedal//dev/bazel/toolchains:extra"
    ],
)

def _headers_filter_impl(ctx):
    all_headers = []
    for dep in ctx.attr.deps:
        all_headers += _collect_headers(dep)
    all_headers = utils.unique_files(all_headers)
    filtered_headers = []
    for header in all_headers:
        if (_match_file_name(header, ctx.attr.include) and
            not _match_file_name(header, ctx.attr.exclude)):
            filtered_headers.append(header)
    return [
        DefaultInfo(files = depset(filtered_headers))
    ]


headers_filter = rule(
    implementation = _headers_filter_impl,
    attrs = {
        "deps": attr.label_list(allow_files=True),
        "include": attr.string_list(),
        "exclude": attr.string_list(),
    },
)

def release_include(hdrs, skip_prefix="", add_prefix=""):
    return (hdrs, add_prefix, skip_prefix)

def release(name, include, lib):
    rule_include = []
    rule_include_prefix = []
    rule_include_skip_prefix = []
    for hdrs, prefix, skip_prefix in include:
        for dep in hdrs:
            rule_include.append(dep)
            rule_include_prefix.append(prefix)
            rule_include_skip_prefix.append(skip_prefix)
    _release(
        name = name,
        include = rule_include,
        include_prefix = rule_include_prefix,
        include_skip_prefix = rule_include_skip_prefix,
        lib = lib,
    )
