#!/bin/bash
#===============================================================================
# Copyright 2014 Intel Corporation
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

TBB_VERSION="2021.10.0"
TBB_URL_ROOT="https://github.com/oneapi-src/oneTBB/releases/download/v${TBB_VERSION}"

arch=$(uname -m) 
if [ "${arch}" == "x86_64" ]; then
  os=$(uname)
  if [ "${os}" = "Linux" ]; then
    TBB_OS=lin
    OS=lnx
    ARCH_EXT=tgz
  elif [ "${os}" = "Darwin" ]; then
    TBB_OS=mac
    OS=mac
    ARCH_EXT=tgz
  elif [[ "${os}" =~ "MSYS" ]]; then
    TBB_OS=win
    OS=win
    ARCH_EXT=zip
  else
    echo "Cannot identify operating system. Try downloading package manually."
    exit 1
  fi

  TBB_PACKAGE="oneapi-tbb-${TBB_VERSION}-${TBB_OS}.${ARCH_EXT}"
  TBB_URL="${TBB_URL_ROOT}/${TBB_PACKAGE}"
  DST=$(dirname "$0")/../__deps/tbb
  mkdir -p "${DST}/${OS}"
  DST=$(cd "${DST}" || exit 1;pwd)

  DOWNLOAD_CODE=1

  if [ ! -d "${DST}/${OS}/bin" ]; then
    if [ -x "$(command -v curl)" ]; then
      echo curl -L -o "${DST}/${TBB_PACKAGE}" "${TBB_URL}"
      if curl -L -o "${DST}/${TBB_PACKAGE}" "${TBB_URL}";
      then
        DOWNLOAD_CODE=0
      fi
    elif [ -x "$(command -v wget)" ]; then
      echo wget -O "${DST}/${TBB_PACKAGE}" "${TBB_URL}"
      if wget -O "${DST}/${TBB_PACKAGE}" "${TBB_URL}";
      then
        DOWNLOAD_CODE=0
      fi
    else
      echo "curl or wget not available"
      exit 1
    fi

    if [ ${DOWNLOAD_CODE} -ne 0 ] || [ ! -e "${DST}/${TBB_PACKAGE}" ]; then
      echo "Download from ${TBB_URL} to ${DST} failed"
      exit 1
    fi

    if [ "${OS}" = "win" ]; then
      echo unzip -d "${DST}/${OS}" "${DST}/${TBB_PACKAGE}"
      unzip -d "${DST}/${OS}" "${DST}/${TBB_PACKAGE}"
      mv "${DST}/${OS}/oneapi-tbb-${TBB_VERSION}" "${DST}/${OS}/tbb"
    else
      echo tar -xvf "${DST}/${TBB_PACKAGE}" -C "${DST}"
      tar -C "${DST}/${OS}" --strip-components=1 -xvf "${DST}/${TBB_PACKAGE}"
    fi
    ls -al "${DST}/${OS}/"
    echo "Downloaded and unpacked oneTBB to ${DST}/${OS}"
  else
    echo "oneTBB is already installed in ${DST}/${OS}"
  fi

else
  sudo apt-get update
  sudo apt-get install build-essential gcc gfortran cmake -y
  git clone https://github.com/oneapi-src/oneTBB.git
  CoreCount=$(lscpu -p | grep -Ev '^#' | wc -l)

  rm -rf __deps/tbb
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
fi