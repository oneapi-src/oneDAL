#!/bin/bash
#===============================================================================
# Copyright 2014-2019 Intel Corporation
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



TBB_URL_ROOT="https://github.com/intel/tbb/releases/download/2019_U9/"
TBB_VERSION="tbb2019_20191006oss"
TBB_TARGET_PLATFORM=linux

os=`uname`
if [ "${os}" = "Linux" ]; then
  TBB_OS=lin
  OS=lnx
elif [ "${os}" = "Darwin" ]; then
  TBB_OS=mac
  OS=mac
else
  echo "Cannot identify operating system. Try downloading package manually."
  exit 1
fi

TBB_PACKAGE="${TBB_VERSION}_${TBB_OS}"
TBB_URL=${TBB_URL_ROOT}${TBB_PACKAGE}.tgz
DST=`dirname $0`/../externals/tbb
mkdir -p ${DST}/${OS}
DST=`cd ${DST};pwd`

if [ ! -d "${DST}/${OS}/bin" ]; then
  if [ -x "$(command -v curl)" ]; then
    echo curl -L -o "${DST}/${TBB_PACKAGE}.tgz" "${TBB_URL}"
    curl -L -o "${DST}/${TBB_PACKAGE}.tgz" "${TBB_URL}"
  elif [ -x "$(command -v wget)" ]; then
    echo wget -O "${DST}/${TBB_PACKAGE}.tgz" "${TBB_URL}"
    wget -O "${DST}/${TBB_PACKAGE}.tgz" "${TBB_URL}"
  else
    echo "curl or wget not available"
    exit 1
  fi

  if [ $? -ne 0 -o ! -e "${DST}/${TBB_PACKAGE}.tgz" ]; then
    echo "Download from ${TBB_URL} to ${DST} failed"
    exit 1
  fi
  
  echo tar -xvf "${DST}/${TBB_PACKAGE}.tgz" -C ${DST}
  tar -C ${DST}/${OS} --strip-components=1 -xvf "${DST}/${TBB_PACKAGE}.tgz" ${TBB_VERSION}
  echo "Downloaded and unpacked Intel(R) TBB to ${DST}/${OS}"
else
  echo "Intel(R) TBB is already installed in ${DST}/${OS}"
fi
