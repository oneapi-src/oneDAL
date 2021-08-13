#===============================================================================
# Copyright 2020-2021 Intel Corporation
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

mpi_repo = repos.prebuilt_libs_repo_rule(
    includes = [
        "include",
    ],
    libs = [
        "lib/release/libmpi.so",
        "lib/release/libmpi.so.12",
        "lib/release/libmpi.so.12.0",
        "lib/release/libmpi.so.12.0.0",
        "libfabric/lib/libfabric.so",
        "libfabric/lib/libfabric.so.1",
    ],
    build_template = "@onedal//dev/bazel/deps:mpi.tpl.BUILD",
    download_mapping = {
        # Required directory layout and layout in the downloaded
        # archives may be different. Mapping helps to setup relations
        # between the required layout (LHS) and downloaded (RHS).
        #          REQUIRED                       DOWNLOADED
        "libfabric/lib/libfabric.so":    "lib/libfabric/libfabric.so",
        "libfabric/lib/libfabric.so.1":  "lib/libfabric/libfabric.so.1",
        "lib/release/libmpi.so":         "lib/libmpi.so",
        "lib/release/libmpi.so.12":      "lib/libmpi.so.12",
        "lib/release/libmpi.so.12.0":    "lib/libmpi.so.12.0",
        "lib/release/libmpi.so.12.0.0":  "lib/libmpi.so.12.0.0",
    },
)
