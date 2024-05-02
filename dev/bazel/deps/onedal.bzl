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

load("@onedal//dev/bazel:repos.bzl", "repos")

onedal_repo = repos.prebuilt_libs_repo_rule(
    includes = [
        "include",
    ],
    libs = [
        # Static
        "lib/libonedal_core.a",
        "lib/libonedal_thread.a",
        "lib/libonedal.a",
        "lib/libonedal_dpc.a",
        "lib/libonedal_sycl.a",
        "lib/libonedal_parameters.a",
        "lib/libonedal_parameters_dpc.a",

        # Dynamic
        "lib/libonedal_core.so",
        "lib/libonedal_thread.so",
        "lib/libonedal.so",
        "lib/libonedal_dpc.so",
        "lib/libonedal_parameters.so",
        "lib/libonedal_parameters_dpc.so",
    ],
    build_template = "@onedal//dev/bazel/deps:onedal.tpl.BUILD",
)
