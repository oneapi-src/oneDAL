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
    "cc_test",
)
load("@onedal//dev/bazel:utils.bzl",
    "sets",
    "paths",
)
load("@onedal//dev/bazel/config:config.bzl",
    "CpuInfo",
)

def dal_public_includes(name, deps=[]):
    _dal_module(
        name = name,
        hdrs = native.glob(["**/*.hpp"], exclude=["backend/**/*"]),
        deps = deps,
    )

def dal_module(name, features=[],
               hdrs=[], srcs=[],
               hdrs_cc=[], srcs_cc=[],
               hdrs_dpc=[], srcs_dpc=[],
               host=True, dpc=False, auto=False,
               public_includes=False, **kwargs):
    if auto:
        hpp_filt = ["**/*.hpp"]
        cpp_filt = ["**/*.cpp"]
        dpc_filt = ["**/*_dpc.cpp"]
        test_filt = ["**/*_test*"]
        hdrs_all = native.glob(hpp_filt, exclude=test_filt)
        hdrs_cc_auto = hdrs_all
        hdrs_dpc_auto = hdrs_all
        srcs_cc_auto = native.glob(cpp_filt, exclude=test_filt + dpc_filt)
        srcs_dpc_auto = native.glob(cpp_filt, exclude=test_filt)
    else:
        hdrs_cc_auto = []
        hdrs_dpc_auto = []
        srcs_cc_auto = []
        srcs_dpc_auto = []
    if host:
        _dal_module(
            name = name,
            features = features,
            hdrs = hdrs_cc_auto + hdrs_cc + hdrs,
            srcs = srcs_cc_auto + srcs_cc + srcs,
            **kwargs,
        )
    if dpc:
        _dal_module(
            name = name + "_dpc",
            features = ["dpc++"] + features,
            hdrs = hdrs_dpc_auto + hdrs_dpc + hdrs,
            srcs = srcs_dpc_auto + srcs_dpc + srcs,
            local_defines = [ "ONEAPI_DAL_DATA_PARALLEL" ],
            **kwargs,
        )


def dal_static_lib(name, lib_name, host=True, dpc=False, deps=[],
                   lib_tags=["dal"], external_deps=[], **kwargs):
    if host:
        cc_static_lib(
            name = name,
            lib_name = lib_name,
            lib_tags = lib_tags,
            deps = deps + external_deps,
            **kwargs
        )
    if dpc:
        cc_static_lib(
            name = name + "_dpc",
            lib_name = lib_name + "_dpc",
            lib_tags = lib_tags,
            deps = [ d + "_dpc" for d in deps ] + external_deps,
            **kwargs
        )

def dal_algos(name, algos):
    algo_labels = []
    algo_labels_dpc = []
    public_includes_labels = []
    for algo in algos:
        algo_label = "@onedal//cpp/oneapi/dal/algo/{0}:{0}".format(algo)
        algo_label_dpc = "@onedal//cpp/oneapi/dal/algo/{0}:{0}_dpc".format(algo)
        public_includes_label = "@onedal//cpp/oneapi/dal/algo/{0}:public_includes".format(algo)
        _dal_module(
            name = algo,
            hdrs = [ "algo/{}.hpp".format(algo) ],
            deps = [ algo_label ],
        )
        _dal_module(
            name = algo + "_dpc",
            hdrs = [ "algo/{}.hpp".format(algo) ],
            deps = [ algo_label_dpc ],
        )
        algo_labels.append(algo_label)
        algo_labels_dpc.append(algo_label_dpc)
        public_includes_labels.append(public_includes_label)
    _dal_module(
        name = name + "_public_includes",
        deps = public_includes_labels,
    )
    _dal_module(
        name = name,
        deps = algo_labels,
    )
    _dal_module(
        name = name + "_dpc",
        deps = algo_labels_dpc,
    )

def dal_test(name, deps=[], test_deps=[], **kwargs):
    _dal_module(
        name = name + "_module",
        deps = select({
            "@config//:dev_test_link_mode": [
                "@onedal//cpp/daal:threading_static",
            ] + deps,
            "@config//:static_test_link_mode": [
                "@onedal//cpp/oneapi/dal:static",
                "@onedal//cpp/daal:core_static",
                "@onedal//cpp/daal:threading_static",
            ],
            "@config//:dynamic_test_link_mode": [
                # TODO
                # ":threading_dynamic",
            ],
            "@config//:release_static_test_link_mode": [
                "@onedal_release//:onedal_static",
                "@onedal_release//:core_static",
                "@onedal//cpp/daal:threading_release_static",
            ],
            "@config//:release_dynamic_test_link_mode": [
                # TODO
                # ":threading_release_dynamic",
            ],
        }) + test_deps,
        **kwargs,
    )
    cc_test(
        name = name,
        deps = [ ":{}_module".format(name) ],
    )

def dal_examples(srcs, non_alg_examples=[]):
    dal_module(
        name = "example_util",
        hdrs = native.glob(["source/example_util/*.hpp"]),
        includes = [ "source" ],
    )
    for src in srcs:
        _, alg_name, src_file = src.rsplit('/', 2)
        example_name, _ = paths.split_extension(src_file)
        if alg_name in non_alg_examples:
            dep = "@onedal//cpp/oneapi/dal:core"
        else:
            dep = "@onedal//cpp/oneapi/dal:{}".format(alg_name)
        dal_test(
            name = example_name,
            srcs = [ src ],
            deps = [ dep ],
            test_deps = [
                ":example_util",
            ],
        )

def _dal_module(name, lib_tag="dal", features=[], **kwargs):
    cc_module(
        name = name,
        lib_tag = lib_tag,
        features = [ "pedantic", "c++17" ] + features,
        disable_mic = True,
        **kwargs,
    )

def _dal_generate_cpu_dispatcher_impl(ctx):
    cpus = sets.make(ctx.attr._cpus[CpuInfo].enabled)
    content = (
        "// DO NOT EDIT: file is auto-generated on build time\n" +
        "// DO NOT PUT THIS FILE TO SVC: file is auto-generated on build time\n" +
        "// CPU detection logic specified in dev/bazel/config.bzl file\n" +
        "\n" +
        ("#define ONEDAL_CPU_DISPATCH_SSSE3\n"      if sets.contains(cpus, "ssse3")      else "") +
        ("#define ONEDAL_CPU_DISPATCH_SSE42\n"      if sets.contains(cpus, "sse42")      else "") +
        ("#define ONEDAL_CPU_DISPATCH_AVX\n"        if sets.contains(cpus, "avx")        else "") +
        ("#define ONEDAL_CPU_DISPATCH_AVX2\n"       if sets.contains(cpus, "avx2")       else "") +
        ("#define ONEDAL_CPU_DISPATCH_AVX512\n"     if sets.contains(cpus, "avx512")     else "")
    )
    kernel_defines = ctx.actions.declare_file(ctx.attr.out)
    ctx.actions.write(kernel_defines, content)
    return [ DefaultInfo(files=depset([ kernel_defines ])) ]

dal_generate_cpu_dispatcher = rule(
    implementation = _dal_generate_cpu_dispatcher_impl,
    output_to_genfiles = True,
    attrs = {
        "out": attr.string(mandatory=True),
        "_cpus": attr.label(
            default = "@config//:cpu",
        ),
    },
)
