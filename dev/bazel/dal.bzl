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
load("@onedal//dev/bazel:cc.bzl", "ModuleInfo")

def dal_module(name, hdrs=[], srcs=[],
               dal_deps=[], extra_deps=[],
               host_hdrs=[], host_srcs=[], host_deps=[],
               dpc_hdrs=[], dpc_srcs=[], dpc_deps=[],
               auto=False, host=True, dpc=True, **kwargs):
    _check_target_name(name, host, dpc)
    if auto:
        hpp_filt = ["**/*.hpp"]
        cpp_filt = ["**/*.cpp"]
        dpc_filt = ["**/*_dpc.cpp"]
        test_filt = ["**/*_test*", "**/test/**"]
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
    _add_module_if_condition(
        condition = host,
        name = _remove_dpc_suffix(name),
        hdrs = hdrs + host_auto_hdrs + host_hdrs,
        srcs = srcs + host_auto_srcs + host_srcs,
        deps = dal_deps + host_deps + extra_deps,
        **kwargs,
    )
    _add_module_if_condition(
        condition = dpc,
        name = _get_dpc_target_name(name, host, dpc),
        hdrs = hdrs + dpc_auto_hdrs + dpc_hdrs,
        srcs = srcs + dpc_auto_srcs + dpc_srcs,
        deps = _get_dpc_deps(dal_deps) + dpc_deps + extra_deps,
        is_dpc = True,
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
            "oneapi/",
        ],
        exclude = [
            "backend/",
            "test/",
            "bazel-",
        ],
    )

def dal_static_lib(name, lib_name, dal_deps=[], host_deps=[],
                   dpc_deps=[], extra_deps=[], lib_tags=["dal"],
                   **kwargs):
    cc_static_lib(
        name = name,
        lib_name = lib_name,
        lib_tags = lib_tags,
        deps = dal_deps + extra_deps + host_deps,
        **kwargs
    )
    cc_static_lib(
        name = _get_dpc_target_name(name, True, True),
        lib_name = _get_dpc_target_name(lib_name, True, True),
        lib_tags = lib_tags,
        deps = _get_dpc_deps(dal_deps) + extra_deps + dpc_deps,
        **kwargs
    )

def dal_dynamic_lib(name, lib_name, dal_deps=[], host_deps=[],
                    dpc_deps=[], extra_deps=[], lib_tags=["dal"],
                    **kwargs):
    cc_dynamic_lib(
        name = name,
        lib_name = lib_name,
        lib_tags = lib_tags,
        deps = dal_deps + extra_deps + host_deps,
        **kwargs
    )
    cc_dynamic_lib(
        name = _get_dpc_target_name(name, True, True),
        lib_name = _get_dpc_target_name(lib_name, True, True),
        lib_tags = lib_tags,
        deps = _get_dpc_deps(dal_deps) + extra_deps + dpc_deps,
        **kwargs
    )

def dal_test(name, hdrs=[], srcs=[],
             dal_deps=[], extra_deps=[],
             host_hdrs=[], host_srcs=[], host_deps=[],
             dpc_hdrs=[], dpc_srcs=[], dpc_deps=[],
             host=True, dpc=True, framework="gtest",
             data=[], tags=[], **kwargs):
    # TODO: Refactor this rule once decision on the tests structure is made
    if not framework in ["gtest"]:
        fail("Unknown test framework '{}' in test rule '{}'".format(framework, name))
    module_name = "__" + name
    if not host and dpc:
        module_name = _remove_dpc_suffix(module_name) + "_dpc"
    dal_module(
        name = module_name,
        hdrs = hdrs,
        srcs = srcs,
        host_hdrs = host_hdrs,
        host_srcs = host_srcs,
        host_deps = host_deps,
        dpc_hdrs = dpc_hdrs,
        dpc_srcs = dpc_srcs,
        dpc_deps = dpc_deps,
        host = host,
        dpc = dpc,
        dal_deps = _select({
            "@config//:dev_test_link_mode": dal_deps,
            "@config//:static_test_link_mode": [
                "@onedal//cpp/oneapi/dal:static",
            ],
            "@config//:dynamic_test_link_mode": [
                "@onedal//cpp/oneapi/dal:dynamic",
            ],
            "@config//:release_static_test_link_mode": [
                "@onedal_release//:onedal_static",
            ],
            "@config//:release_dynamic_test_link_mode": [
                "@onedal_release//:onedal_dynamic",
            ],
        }) + [
            "@onedal//cpp/oneapi:include_root",
            "@onedal//cpp/oneapi/dal/test/engine:common",
            "@onedal//cpp/oneapi/dal/test/engine:main",
        ],
        extra_deps = _select({
            "@config//:dev_test_link_mode": [
                "@onedal//cpp/daal:threading_static",
            ],
            "@config//:static_test_link_mode": [
                "@onedal//cpp/daal:core_static",
                "@onedal//cpp/daal:threading_static",
            ],
            "@config//:dynamic_test_link_mode": [
                "@onedal//cpp/daal:core_dynamic",
                "@onedal//cpp/daal:threading_dynamic",
            ],
            "@config//:release_static_test_link_mode": [
                "@onedal_release//:core_static",
                "@onedal//cpp/daal:threading_release_static",
            ],
            "@config//:release_dynamic_test_link_mode": [
                "@onedal_release//:core_dynamic",
                "@onedal//cpp/daal:threading_release_dynamic",
            ],
        }) + extra_deps,
        testonly = True,
        **kwargs,
    )
    if host:
        cc_test(
            name = name,
            deps = [ ":" + module_name ],
            data = data,
            tags = tags + ["host"],
        )
    if dpc:
        cc_test(
            name = _get_dpc_target_name(name, host, dpc),
            features = [ "dpc++" ],
            deps = [
                ":" + _get_dpc_target_name(module_name, host, dpc),
                "@opencl//:opencl_binary",
            ],
            data = data,
            tags = tags + ["dpc"],
        )

def dal_test_suite(name, srcs=[], tests=[],
                   host=True, dpc=True, **kwargs):
    _check_target_name(name, host, dpc)
    targets = []
    targets_dpc = []
    for test_file in srcs:
        target = test_file.replace(".cpp", "").replace("/", "_")
        dal_test(
            name = target,
            srcs = [test_file],
            host = host,
            dpc = dpc,
            **kwargs,
        )
        if host:
            targets.append(":" + target)
        if dpc:
            targets_dpc.append(":" + _get_dpc_target_name(target, host, dpc))
    native.test_suite(
        name = name,
        tests = tests + targets + targets_dpc,
    )

def dal_collect_test_suites(name, root, modules, tests=[], **kwargs):
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
        framework = "none",
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

def dal_global_header_test(name, algo_dir, algo_exclude=[], algo_preview=[], dal_deps=[]):
    _generate_global_header_test_cpp(
        name = "_" + name,
        algo_dir = algo_dir,
        algo_exclude = algo_exclude,
        algo_preview = algo_preview,
        deps = dal_deps,
    )
    dal_test(
        name = name,
        srcs = [
            ":_" + name,
        ],
        dal_deps = dal_deps,
    )

def _collect_algorithm_names(algo_dir, deps):
    algo_names = []
    for dep in deps:
        if not ModuleInfo in dep:
            continue
        headers = dep[ModuleInfo].compilation_context.headers.to_list()
        for header in headers:
            header_path = header.path
            header_name = paths.basename(header_path)
            if (algo_dir + "/" + header_name) in header_path:
                algo_name, _ = paths.split_extension(header_name)
                algo_names.append(algo_name)
    return algo_names

def _generate_global_header_test_cpp_impl(ctx):
    algo_dir = paths.normalize(ctx.attr.algo_dir)
    algo_names = _collect_algorithm_names(algo_dir, ctx.attr.deps)
    filtered_algo_names = sets.to_list(
        sets.difference(sets.make(algo_names),
                        sets.make(ctx.attr.algo_exclude))
    )
    preview_algo_names = sets.make(ctx.attr.algo_preview)
    test_file = ctx.actions.declare_file("{}.cpp".format(ctx.label.name))
    content = "#include \"oneapi/dal.hpp\"\n"
    # Add comments for people who may open this file to understand root cause of an error
    content += (
        "\n" +
        "// This is automatically generated file for testing `dal.hpp`.\n" +
        "// If you encountered error compiling this file, it means not all the algorithms\n" +
        "// declared in `{}` are included to `dal.hpp`. Make sure all algorithm are included\n".format(algo_dir) +
        "// or add exception to `algo_exclude` list in `{}`.\n".format(ctx.build_file_path) +
        "\n"
    )
    for algo_name in filtered_algo_names:
        if sets.contains(preview_algo_names, algo_name):
            content += "using namespace oneapi::dal::preview::{};\n".format(algo_name)
        else:
            content += "using namespace oneapi::dal::{};\n".format(algo_name)
    # Linkers may complain about empty object files, so
    # add dummy function. It's more safe to declare static function, but
    # compiler warns about unused functions in this case
    content += "\nvoid __headers_test_dummy__() {}\n"
    ctx.actions.write(test_file, content)
    return [ DefaultInfo(files=depset([ test_file ])) ]

_generate_global_header_test_cpp = rule(
    implementation = _generate_global_header_test_cpp_impl,
    output_to_genfiles = True,
    attrs = {
        "algo_dir": attr.string(mandatory=True),
        "algo_exclude": attr.string_list(),
        "algo_preview": attr.string_list(),
        "deps": attr.label_list(mandatory=True),
    },
)

def _dal_module(name, lib_tag="dal", is_dpc=False, features=[],
                local_defines=[], deps=[], **kwargs):
    cc_module(
        name = name,
        lib_tag = lib_tag,
        features = [ "pedantic", "c++17" ] + features + (
            ["dpc++"] if is_dpc else []
        ),
        disable_mic = True,
        cpu_defines = {
            "sse2":   [ "__CPU_TAG__=oneapi::dal::backend::cpu_dispatch_sse2"   ],
            "ssse3":  [ "__CPU_TAG__=oneapi::dal::backend::cpu_dispatch_ssse3"  ],
            "sse42":  [ "__CPU_TAG__=oneapi::dal::backend::cpu_dispatch_sse42"  ],
            "avx":    [ "__CPU_TAG__=oneapi::dal::backend::cpu_dispatch_avx"    ],
            "avx2":   [ "__CPU_TAG__=oneapi::dal::backend::cpu_dispatch_avx2"   ],
            "avx512": [ "__CPU_TAG__=oneapi::dal::backend::cpu_dispatch_avx512" ],
        },
        local_defines = local_defines + ([
            "DAAL_SYCL_INTERFACE",
            "ONEDAL_DATA_PARALLEL"
        ] if is_dpc else []),
        deps = _expand_select(deps),
        **kwargs,
    )

def _add_module_if_condition(condition, name, is_dpc=False, **kwargs):
    if condition:
        _dal_module(name=name, is_dpc=is_dpc, **kwargs)
    else:
        _dal_module(name=name, is_dpc=is_dpc)

def _select(x):
    return [x]

def _normalize_dep(dep):
    if dep.rfind(":") > 0:
        return dep
    last_slash_index = dep.rfind("/")
    if last_slash_index < 0:
        return dep
    package_name = dep[last_slash_index + 1:].rstrip()
    return dep + ":" + package_name

def _get_dpc_dep_name(name):
    return _normalize_dep(name) + "_dpc"

def _get_dpc_deps(deps):
    normalized = []
    for dep in deps:
        if type(dep) == "dict":
            normalized_dep = {}
            for key, value in dep.items():
                normalized_dep[key] = [
                    _get_dpc_dep_name(x) for x in value
                ]
            normalized.append(normalized_dep)
        else:
            normalized.append(_get_dpc_dep_name(dep))
    return normalized

def _expand_select(deps):
    expanded = []
    for dep in deps:
        if type(dep) == 'dict':
            expanded += select(dep)
        else:
            expanded += [dep]
    return expanded

def _check_target_name(name, host, dpc, suffix="_dpc"):
    if not (host or dpc):
        fail("Target must specify at leat one of " +
             "`host = True` or `dpc = True`")
    if not host and dpc and not name.endswith(suffix):
        fail("Target names that defined with `host = False` " +
             "and `dpc = True` must end with `{}` suffix".format(suffix))

def _get_dpc_target_name(name, host, dpc, suffix="_dpc"):
    return name + (suffix if host else "")

def _remove_dpc_suffix(name, suffix="_dpc"):
    index = name.rfind(suffix)
    if index > 0:
        return name[:index + 1]
    else:
        return name

def _check_test_target_name(name, host, dpc, suffix="_dpc_test"):
    _check_target_name(name, host, dpc, suffix)

def _get_dpc_test_target_name(name, host, dpc, suffix="_dpc_test"):
    return _get_dpc_target_name(name, host, dpc, suffix)

def _remove_dpc_test_suffix(name, suffix="_dpc_test"):
    return _remove_dpc_suffix(name, suffix)
