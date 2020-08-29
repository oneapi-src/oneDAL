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
    "cc_test",
)
load("@onedal//dev/bazel:release.bzl",
    "headers_filter",
)
load("@onedal//dev/bazel:utils.bzl",
    "sets",
    "utils",
    "paths",
)
load("@onedal//dev/bazel/config:config.bzl",
    "CpuInfo",
)

def dal_module(name, features=[], hdrs=[], srcs=[],
               dal_deps=[], extra_deps=[],
               host_hdrs=[], host_srcs=[], host_deps=[],
               dpc_hdrs=[], dpc_srcs=[], dpc_deps=[],
               auto=False, host=True, dpc=True,
               local_defines=[], **kwargs):
    if auto:
        hpp_filt = ["**/*.hpp"]
        cpp_filt = ["**/*.cpp"]
        dpc_filt = ["**/*_dpc.cpp"]
        test_filt = ["**/*_test*"]
        hdrs_all = native.glob(hpp_filt, exclude=test_filt)
        dpc_auto_hdrs = hdrs_all
        dpc_auto_srcs = native.glob(cpp_filt, exclude=test_filt)
        host_auto_hdrs = hdrs_all
        host_auto_srcs = native.glob(cpp_filt, exclude=test_filt + dpc_filt)
    else:
        host_auto_hdrs = []
        host_auto_srcs = []
        dpc_auto_hdrs = []
        dpc_auto_srcs = []
    if host:
        _dal_module(
            name = name,
            features = features,
            hdrs = hdrs + host_auto_hdrs + host_hdrs,
            srcs = srcs + host_auto_srcs + host_srcs,
            deps = dal_deps + host_deps + extra_deps,
            local_defines = local_defines,
            **kwargs,
        )
    if dpc:
        _dal_module(
            name = name + "_dpc",
            features = ["dpc++"] + features,
            hdrs = hdrs + dpc_auto_hdrs + dpc_hdrs,
            srcs = srcs + dpc_auto_srcs + dpc_srcs,
            deps = _get_dpc_deps(dal_deps) + dpc_deps + extra_deps,
            local_defines = local_defines + [ "ONEAPI_DAL_DATA_PARALLEL" ],
            **kwargs,
        )

def dal_collect_modules(name, root, modules, dal_deps=[], **kwargs):
    module_deps = []
    for module_name in modules:
        module_label = "{0}/{1}".format(root, module_name)
        dal_module(
            name = module_name,
            hdrs = native.glob(["{}*.hpp".format(module_name)]),
            dal_deps = [ module_label ],
        )
        module_deps.append(":" + module_name)
    dal_module(
        name = name,
        dal_deps = dal_deps + module_deps,
        **kwargs,
    )

def dal_public_includes(name, dal_deps=[], **kwargs):
    headers_filter(
        name = name,
        deps = dal_deps + _get_dpc_deps(dal_deps),
        include = [
            "oneapi/dal/",
        ],
        exclude = [
            "backend/",
            "test/",
            "bazel-",
        ],
    )

def dal_static_lib(name, lib_name, dal_deps=[], host_deps=[],
                   dpc_deps=[], extra_deps=[], lib_tags=["dal"], **kwargs):
    cc_static_lib(
        name = name,
        lib_name = lib_name,
        lib_tags = lib_tags,
        deps = dal_deps + extra_deps + host_deps,
        **kwargs
    )
    cc_static_lib(
        name = name + "_dpc",
        lib_name = lib_name + "_dpc",
        lib_tags = lib_tags,
        deps = _get_dpc_deps(dal_deps) + extra_deps + dpc_deps,
        **kwargs
    )

def dal_dynamic_lib(name, lib_name, dal_deps=[], host_deps=[],
                   dpc_deps=[], extra_deps=[], lib_tags=["dal"], **kwargs):
    cc_dynamic_lib(
        name = name,
        lib_name = lib_name,
        lib_tags = lib_tags,
        deps = dal_deps + extra_deps + host_deps,
        **kwargs
    )
    cc_dynamic_lib(
        name = name + "_dpc",
        lib_name = lib_name + "_dpc",
        lib_tags = lib_tags,
        deps = _get_dpc_deps(dal_deps) + extra_deps + dpc_deps,
        **kwargs
    )

def dal_test(name, dal_deps=[], test_deps=[], data=[],
             gtest=True, tags=[], **kwargs):
    # TODO: Add support for DPC++
    _dal_module(
        name = "_" + name,
        deps = select({
            "@config//:dev_test_link_mode": [
                "@onedal//cpp/daal:threading_static",
            ] + dal_deps,
            "@config//:static_test_link_mode": [
                "@onedal//cpp/oneapi/dal:static",
                "@onedal//cpp/daal:core_static",
                "@onedal//cpp/daal:threading_static",
            ],
            "@config//:dynamic_test_link_mode": [
                "@onedal//cpp/oneapi/dal:dynamic",
                "@onedal//cpp/daal:core_dynamic",
                "@onedal//cpp/daal:threading_dynamic",
            ],
            "@config//:release_static_test_link_mode": [
                "@onedal_release//:onedal_static",
                "@onedal_release//:core_static",
                "@onedal//cpp/daal:threading_release_static",
            ],
            "@config//:release_dynamic_test_link_mode": [
                "@onedal_release//:onedal_dynamic",
                "@onedal_release//:core_dynamic",
                "@onedal//cpp/daal:threading_release_dynamic",
            ],
        }) + test_deps + ([
            "@gtest//:gtest_main",
        ] if gtest else []),
        **kwargs,
    )
    cc_test(
        name = name,
        deps = [ ":_" + name ],
        data = data,
        tags = tags,
    )

def dal_test_suite(name, srcs=[], tests=[], **kwargs):
    targets = []
    for test_file in srcs:
        target = test_file.replace(".cpp", "").replace("/", "_")
        dal_test(
            name = target,
            srcs = [test_file],
            **kwargs,
        )
        targets.append(":" + target)
    native.test_suite(
        name = name,
        tests = tests + targets,
    )

def dal_collect_tests(name, root, modules, tests=[], **kwargs):
    test_deps = []
    for module_name in modules:
        test_label = "{0}/{1}:tests".format(root, module_name)
        test_deps.append(test_label)
    dal_test_suite(
        name = name,
        tests = tests + test_deps,
        **kwargs,
    )

def dal_example(name, dal_deps=[], **kwargs):
    dal_test(
        name = name,
        dal_deps = [
            "@onedal//cpp/oneapi/dal:core",
            "@onedal//cpp/oneapi/dal/io",
        ] + dal_deps,
        gtest = False,
        **kwargs,
    )

def dal_example_suite(name, srcs, **kwargs):
    suite_deps = []
    for src in srcs:
        _, alg_name, src_file = src.rsplit('/', 2)
        example_name, _ = paths.split_extension(src_file)
        dal_example(
            name = example_name,
            srcs = [ src ],
            **kwargs,
        )
        suite_deps.append(":" + example_name)
    native.test_suite(
        name = name,
        tests = suite_deps,
    )

def dal_algo_example_suite(algos, dal_deps=[], **kwargs):
    for algo in algos:
        dal_example_suite(
            name = algo,
            srcs = native.glob(["source/{}/*.cpp".format(algo)]),
            dal_deps = dal_deps + [
                "@onedal//cpp/oneapi/dal/algo:{}".format(algo),
            ],
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

def _dal_module(name, lib_tag="dal", features=[], **kwargs):
    cc_module(
        name = name,
        lib_tag = lib_tag,
        features = [ "pedantic", "c++17" ] + features,
        disable_mic = True,
        cpu_defines = {
            "sse2":   [ "__CPU_TAG__=oneapi::dal::backend::cpu_dispatch_default" ],
            "ssse3":  [ "__CPU_TAG__=oneapi::dal::backend::cpu_dispatch_ssse3"   ],
            "sse42":  [ "__CPU_TAG__=oneapi::dal::backend::cpu_dispatch_sse42"   ],
            "avx":    [ "__CPU_TAG__=oneapi::dal::backend::cpu_dispatch_avx"     ],
            "avx2":   [ "__CPU_TAG__=oneapi::dal::backend::cpu_dispatch_avx2"    ],
            "avx512": [ "__CPU_TAG__=oneapi::dal::backend::cpu_dispatch_avx512"  ],
        },
        **kwargs,
    )

def _normalize_dep(dep):
    if dep.rfind(":") > 0:
        return dep
    last_slash_index = dep.rfind("/")
    if last_slash_index < 0:
        return dep
    package_name = dep[last_slash_index + 1:].rstrip()
    return dep + ":" + package_name

def _normalize_deps(deps):
    return [_normalize_dep(x) for x in deps]

def _get_dpc_deps(deps):
    normalized_deps = _normalize_deps(deps)
    return [x + "_dpc" for x in normalized_deps]
