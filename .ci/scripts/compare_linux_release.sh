#!/bin/bash
#===============================================================================
# Copyright contributors to the oneDAL project
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

# Compares the Make and Bazel Linux releases for the nightly Linux job.
# Nightly-test runs main's workflow against the tested checkout, so the
# invocation lives here, versioned with the files and flags it names.
# Usage: compare_linux_release.sh [make_release_dir] [bazel_release_dir]
# (relative to the repo root; defaults are the nightly job's layout).

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

make_release_dir=${1:-__release_lnx/daal/latest}
bazel_release_dir=${2:-bazel-bin/release/daal/latest}

# Self-check the undefined-symbol filter first, so a widened prefix fails here
# rather than letting the releases compare equal for the wrong reason.
python dev/release_tests/compare_release_linkage.py

python dev/release_tests/compare_release_trees.py \
    --platform linux \
    --make "${make_release_dir}" \
    --bazel "${bazel_release_dir}" \
    --check-level 4 \
    --make-symlinks-dereferenced \
    --cross-toolchain
