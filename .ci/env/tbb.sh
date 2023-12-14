#!/bin/bash
#===============================================================================
# Copyright 2023-24 FUJITSU LIMITED
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
sudo apt-get install build-essential gcc gfortran cmake
git clone https://github.com/oneapi-src/oneTBB.git
CoreCount=$(lscpu -p | grep -Ev '^#' | wc -l)
pushd oneTBB
  git checkout v2021.11.0
  mkdir build
  pushd build
    cmake -DCMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=Release -DTBB_TEST=OFF -DTBB_STRICT_PROTOTYPES=OFF -DCMAKE_INSTALL_PREFIX=../../__deps/tbb  .. 
    make -j${CoreCount} 
    make install
  popd
popd
rm -rf oneTBB

pushd __deps/tbb
    mkdir -p lnx
    mv lib/ lnx/
    mv include/ lnx/ 
    pushd lnx
        mkdir -p lib/arm/gcc4.8
        mv lib/libtbb* lib/arm/gcc4.8
    popd
popd 
