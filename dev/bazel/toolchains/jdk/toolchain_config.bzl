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

def _extra_jdk_tools_impl(ctx):
    toolchain_info = platform_common.ToolchainInfo(
        extract_jni_headers = ctx.attr.extract_jni_headers,
    )
    return [toolchain_info]

extra_jdk_tools = rule(
    implementation = _extra_jdk_tools_impl,
    attrs = {
        "extract_jni_headers": attr.string(mandatory=True),
    },
)
