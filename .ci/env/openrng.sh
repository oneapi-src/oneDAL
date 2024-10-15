#!/bin/bash
#===============================================================================
# Copyright contributors to the oneDAL project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

SCRIPT_PATH=$(readlink -f "${BASH_SOURCE[0]}")
SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")
ONEDAL_DIR=$(readlink -f "${SCRIPT_DIR}/../..")
OPENRNG_DEFAULT_SOURCE_DIR="${ONEDAL_DIR}/__work/openrng"

show_help() {
  echo "Usage: $0 [--help]"
  column -t -s":" <<< '--help:Display this information
--rng-src:The path to an existing OpenRNG source dircetory. The source is cloned if this parameter is omitted
--prefix:The path where OpenRNG will be installed
'
}

while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
        --rng-src)
        rng_src_dir="$2"
        shift;;
        --prefix)
        PREFIX="$2"
        shift;;
        --help)
        show_help
        exit 0
        ;;
        *)
        echo "Unknown option: $1"
        exit 1
        ;;
    esac
    shift
done

OPENRNG_DEFAULT_PREFIX="${ONEDAL_DIR}"/__deps/openrng
rng_prefix="${PREFIX:-${OPENRNG_DEFAULT_PREFIX}}"

rng_src_dir=${rng_src_dir:-$OPENRNG_DEFAULT_SOURCE_DIR}
if [[ ! -d "${rng_src_dir}" ]] ; then
    git clone https://git.gitlab.arm.com/libraries/openrng.git "${rng_src_dir}"
fi

echo $rng_src_dir
echo $rng_prefix
pushd "${rng_src_dir}"
    rm -rf build
    mkdir build
        pushd build
            cmake -DCMAKE_INSTALL_PREFIX="${rng_prefix}" ./../
            make install
        popd
popd
