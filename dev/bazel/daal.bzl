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
    "cc_depset",
    "cc_static_lib",
)
load("@onedal//dev/bazel:utils.bzl",
    "sets",
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

def daal_depset(**kwargs):
    cc_depset(**kwargs)

def daal_static_lib(name, lib_tags=["daal"], **kwargs):
    cc_static_lib(
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
        _daal_license_header(version.basename) +
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

def _daal_generate_kernel_defines_impl(ctx):
    cpus = sets.make(ctx.attr._cpus[CpuInfo].enabled)
    kernel_defines = ctx.actions.declare_file(ctx.attr.out)
    content = (
        _daal_license_header(kernel_defines.basename) +
        "// DO NOT EDIT: file is auto-generated on build time\n" +
        "// DO NOT PUT THIS FILE TO SVC: file is auto-generated on build time\n" +
        "// CPU detection logic specified in dev/bazel/config.bzl file\n" +
        "\n" +
        ("#define DAAL_KERNEL_SSSE3\n"      if sets.contains(cpus, "ssse3")      else "") +
        ("#define DAAL_KERNEL_SSE42\n"      if sets.contains(cpus, "sse42")      else "") +
        ("#define DAAL_KERNEL_AVX\n"        if sets.contains(cpus, "avx")        else "") +
        ("#define DAAL_KERNEL_AVX2\n"       if sets.contains(cpus, "avx2")       else "") +
        ("#define DAAL_KERNEL_AVX512_MIC\n" if sets.contains(cpus, "avx512_mic") else "") +
        ("#define DAAL_KERNEL_AVX512\n"     if sets.contains(cpus, "avx512")     else "")
    )
    ctx.actions.write(kernel_defines, content)
    return [ DefaultInfo(files=depset([ kernel_defines ])) ]

daal_generate_kernel_defines = rule(
    implementation = _daal_generate_kernel_defines_impl,
    output_to_genfiles = True,
    attrs = {
        "out": attr.string(mandatory=True),
        "_cpus": attr.label(
            default = "@config//:cpu",
        ),
    },
)

def _daal_license_header(filename):
    return (
        "/* file: {} */\n".format(filename) +
        "/*******************************************************************************\n" +
        "* Copyright Intel Corporation\n" +
        "*\n" +
        "* Licensed under the Apache License, Version 2.0 (the \"License\");\n" +
        "* you may not use this file except in compliance with the License.\n" +
        "* You may obtain a copy of the License at\n" +
        "*\n" +
        "*     http://www.apache.org/licenses/LICENSE-2.0\n" +
        "*\n" +
        "* Unless required by applicable law or agreed to in writing, software\n" +
        "* distributed under the License is distributed on an \"AS IS\" BASIS,\n" +
        "* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n" +
        "* See the License for the specific language governing permissions and\n" +
        "* limitations under the License.\n" +
        "*******************************************************************************/\n" +
        "\n"
    )
