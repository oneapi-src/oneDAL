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

load("@onedal//dev/bazel:cc.bzl",
    "cc_module",
    "cc_static_lib",
    "cc_dynamic_lib",
)
load("@onedal//dev/bazel:java.bzl",
    "java_jni_headers",
)
load("@onedal//dev/bazel:utils.bzl",
    "sets",
    "paths",
)
load("@onedal//dev/bazel/config:config.bzl",
    "CpuInfo",
    "VersionInfo",
)

def daal_module(name, features=[], lib_tag="daal",
                hdrs=[], srcs=[], auto=False,
                opencl=False, **kwargs):
    if auto:
        auto_hdrs = native.glob(["**/*.h", "**/*.i"])
        auto_srcs = native.glob(["**/*.cpp"])
        if opencl:
            auto_hdrs += native.glob(["**/*.cl"])
    else:
        auto_hdrs = []
        auto_srcs = []
    cc_module(
        name = name,
        lib_tag = lib_tag,
        features = [ "c++11" ] + features,
        cpu_defines = {
            "sse2":       [ "DAAL_CPU=sse2"       ],
            "ssse3":      [ "DAAL_CPU=ssse3"      ],
            "sse42":      [ "DAAL_CPU=sse42"      ],
            "avx":        [ "DAAL_CPU=avx"        ],
            "avx2":       [ "DAAL_CPU=avx2"       ],
            "avx512_mic": [ "DAAL_CPU=avx512_mic" ],
            "avx512":     [ "DAAL_CPU=avx512"     ],
        },
        fpt_defines = {
            "f32": [ "DAAL_FPTYPE=float"  ],
            "f64": [ "DAAL_FPTYPE=double" ],
        },
        hdrs = auto_hdrs + hdrs,
        srcs = auto_srcs + srcs,
        **kwargs,
    )

def daal_static_lib(name, lib_tags=["daal"], **kwargs):
    cc_static_lib(
        name = name,
        lib_tags = lib_tags,
        **kwargs,
    )

def daal_dynamic_lib(name, lib_tags=["daal"], **kwargs):
    cc_dynamic_lib(
        name = name,
        lib_tags = lib_tags,
        **kwargs,
    )

def daal_algorithms(name, algorithms=[]):
    alg_labels = []
    for alg_name in algorithms:
        label = "@onedal//cpp/daal/src/algorithms/{}:kernel".format(alg_name)
        alg_labels.append(label)
    cc_module(
        name = name,
        deps = alg_labels,
    )

def _daal_generate_version_impl(ctx):
    vi = ctx.attr._version_info[VersionInfo]
    version = ctx.actions.declare_file(ctx.attr.out)
    content = (
        "// DO NOT EDIT: file is auto-generated on build time\n" +
        "// DO NOT PUT THIS FILE TO SVC: file is auto-generated on build time\n" +
        "// Product version is specified in dev/bazel/config.bzl file\n" +
        "\n" +
        "#define MAJORVERSION {}\n".format(vi.major) +
        "#define MINORVERSION {}\n".format(vi.minor) +
        "#define UPDATEVERSION {}\n".format(vi.update) +
        "#define BUILD \"{}\"\n".format(vi.build) +
        "#define BUILD_REV \"{}\"\n".format(vi.buildrev) +
        "#define PRODUCT_STATUS '{}'\n".format(vi.status)
    )
    ctx.actions.write(version, content)
    return [ DefaultInfo(files=depset([ version ])) ]

daal_generate_version = rule(
    implementation = _daal_generate_version_impl,
    output_to_genfiles = True,
    attrs = {
        "out": attr.string(mandatory=True),
        "_version_info": attr.label(
            default = "@config//:version",
        ),
    },
)

def daal_java_module(name, **kwargs):
    native.java_library(
        name = name,
        **kwargs
    )

def daal_java_jni_module(name, java_deps=[], java_srcs=[],
                         cc_srcs=[], cc_hdrs=[], cc_deps=[]):
    name_java = name + "_java"
    name_jni_headers = name + "_jni_headers"
    name_cc = name + "_cc"
    daal_java_module(
        name = name_java,
        srcs = java_srcs,
        deps = java_deps,
    )
    java_jni_headers(
        name = name_jni_headers,
        srcs = java_srcs,
        deps = [
            ":" + name_java,
        ],
        out_dir_name = "jni",
    )
    _daal_jni_cc_module(
        name = name_cc,
        includes = ["jni"],
        hdrs = cc_hdrs + [
            ":" + name_jni_headers,
        ],
        srcs = cc_srcs,
        deps = cc_deps + [
            "@onedal_jdk_toolchain//:jni_headers",
        ],
    )

def daal_jni_dynamic_lib(name, lib_tags=["daal_java_api"], **kwargs):
    daal_dynamic_lib(
        name = name,
        lib_tags = lib_tags,
        **kwargs
    )

def daal_jar(*args):
    pass

def _daal_patch_kernel_defines_impl(ctx):
    disabled_cpus = _get_disabled_cpus(ctx)
    kernel_defines = _declare_patched_kernel_defines(ctx)
    ctx.actions.run(
        executable = _get_tool_for_kernel_defines_patching(ctx),
        arguments = [
            ctx.file.src.path,
            kernel_defines.path,
            " ".join(sets.to_list(disabled_cpus)),
        ],
        inputs = [ctx.file.src],
        outputs = [kernel_defines],
    )
    return [ DefaultInfo(files=depset([ kernel_defines ])) ]

daal_patch_kernel_defines = rule(
    implementation = _daal_patch_kernel_defines_impl,
    output_to_genfiles = True,
    attrs = {
        "src": attr.label(allow_single_file=True, mandatory=True),
        "_cpus": attr.label(
            default = "@config//:cpu",
        ),
    },
    toolchains = ["@onedal//dev/bazel/toolchains/extra"],
)


def _daal_jni_cc_module(name, lib_tag="daal_java_api", **kwargs):
    daal_module(
        name = name,
        lib_tag = lib_tag,
        **kwargs
    )

def _get_tool_for_kernel_defines_patching(ctx):
    return ctx.toolchains["@onedal//dev/bazel/toolchains/extra"] \
        .extra_toolchain_info.patch_daal_kernel_defines

def _get_disabled_cpus(ctx):
    cpu_info = ctx.attr._cpus[CpuInfo]
    all_cpus = sets.make(cpu_info.allowed)
    enabled_cpus = sets.make(cpu_info.enabled)
    return sets.difference(all_cpus, enabled_cpus)

def _declare_patched_kernel_defines(ctx):
    relpath = paths.dirname(ctx.build_file_path)
    patched_path = paths.relativize(ctx.file.src.path, relpath)
    return ctx.actions.declare_file(patched_path)
