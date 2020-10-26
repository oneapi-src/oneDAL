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

ExtraToolchainInfo = provider(
    fields = [
        "patch_daal_kernel_defines",
    ],
)

def _extra_toolchain_impl(ctx):
    toolchain_info = platform_common.ToolchainInfo(
        extra_toolchain_info = ExtraToolchainInfo(
            patch_daal_kernel_defines = ctx.attr.patch_daal_kernel_defines,
        ),
    )
    return [toolchain_info]

extra_toolchain = rule(
    implementation = _extra_toolchain_impl,
    attrs = {
        "patch_daal_kernel_defines": attr.string(mandatory=True),
    },
)
