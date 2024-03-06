#!/bin/bash
#===============================================================================
# Copyright 2023 Intel Corporation
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

while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
        --target)
        target="$2"
        ;;
        --compiler)
        compiler="$2"
        ;;
        --host_compiler)
        host_compiler="$2"
        ;;
        --cflags)
        cflags="$2"
        ;;
        --cross_compile)
        cross_compile="$2"
        ;;
        *)
        echo "Unknown option: $1"
        exit 1
        ;;
    esac
    shift
    shift
done

sudo apt-get update
sudo apt-get -y install build-essential gcc gfortran
git clone https://github.com/xianyi/OpenBLAS.git
CoreCount=$(lscpu -p | grep -Ev '^#' | wc -l)
pushd OpenBLAS
  make clean
  if [ "${cross_compile}" == "yes" ]; then
    echo make -j${CoreCount} TARGET=${target} HOSTCC=${host_compiler} CC=${compiler} NO_FORTRAN=1 USE_OPENMP=0 CFLAGS=\"${cflags}\"
    make -j${CoreCount} TARGET=${target} HOSTCC=${host_compiler} CC=${compiler} NO_FORTRAN=1 USE_OPENMP=0 CFLAGS=\"${cflags}\"
  else
    make -j${CoreCount} NO_FORTRAN=1 USE_OPENMP=0
  fi
  make install PREFIX=../__deps/open_blas
popd
