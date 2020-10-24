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

load("@onedal//dev/bazel/toolchains/cc:common.bzl", "detect_os", "detect_compiler")
load("@onedal//dev/bazel/toolchains/cc:toolchain_lnx.bzl", "configure_cc_toolchain_lnx")

def _detect_requirements(repo_ctx):
    os_id = detect_os(repo_ctx)
    compiler_id = detect_compiler(repo_ctx, os_id)
    return struct(
        os_id = os_id,
        compiler_id = compiler_id,

        libc_version = "local",
        libc_abi_version = "local",
        compiler_abi_version = "local",

        host_arch_id = "intel64",
        target_arch_id = "intel64",

        # TODO: Detect compiler version
        compiler_version = "local",

        # TODO: Detect DPC++ compiler, use $env{DPCC}
        dpc_compiler_id = "dpcpp",

        # TODO: Detect compiler version
        dpc_compiler_version = "local",
    )

def _configure_cc_toolchain(repo_ctx, reqs):
    configure_cc_toolchain_os = {
        "lnx": configure_cc_toolchain_lnx,
    }[reqs.os_id]
    return configure_cc_toolchain_os(repo_ctx, reqs)

def _onedal_cc_toolchain_impl(repo_ctx):
    reqs = _detect_requirements(repo_ctx)
    _configure_cc_toolchain(repo_ctx, reqs)

onedal_cc_toolchain = repository_rule(
    implementation = _onedal_cc_toolchain_impl,
    environ = [
        "CC",
        "PATH",
        "INCLUDE",
        "LIB",
    ],
)

def declare_onedal_cc_toolchain(name):
    onedal_cc_toolchain(name = name)
    native.register_toolchains("@{}//:all".format(name))
