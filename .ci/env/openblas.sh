#!/bin/bash
#===============================================================================
# Copyright 2023 Intel Corporation
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

sudo apt-get update
sudo apt-get -y install build-essential gcc gfortran
git clone https://github.com/xianyi/OpenBLAS.git
CoreCount=$(lscpu -p | grep -Ev '^#' | wc -l)
pushd OpenBLAS
  make clean
  make -j${CoreCount} NO_FORTRAN=1
  make install PREFIX=../__deps/open_blas
popd
