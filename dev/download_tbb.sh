#!/bin/bash
#===============================================================================
# Copyright 2014-2020 Intel Corporation
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

TBB_URL_ROOT="https://github.com/oneapi-src/oneTBB/releases/download/v2021.1-beta10/"
TBB_VERSION="oneapi-tbb-2021.1-beta10"

os=$(uname)
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

TBB_PACKAGE="${TBB_VERSION}-${TBB_OS}"
TBB_URL=${TBB_URL_ROOT}${TBB_PACKAGE}.tgz
DST=$(dirname "$0")/../__deps/tbb
mkdir -p "${DST}/${OS}"
DST=$(cd "${DST}" || exit 1;pwd)

DOWNLOAD_CODE=1

if [ ! -d "${DST}/${OS}/bin" ]; then
  if [ -x "$(command -v curl)" ]; then
    echo curl -L -o "${DST}/${TBB_PACKAGE}.tgz" "${TBB_URL}"
    if curl -L -o "${DST}/${TBB_PACKAGE}.tgz" "${TBB_URL}";
    then
      DOWNLOAD_CODE=0
    fi
  elif [ -x "$(command -v wget)" ]; then
    echo wget -O "${DST}/${TBB_PACKAGE}.tgz" "${TBB_URL}"
    if wget -O "${DST}/${TBB_PACKAGE}.tgz" "${TBB_URL}";
    then
      DOWNLOAD_CODE=0
    fi
  else
    echo "curl or wget not available"
    exit 1
  fi

  if [ ${DOWNLOAD_CODE} -ne 0 ] || [ ! -e "${DST}/${TBB_PACKAGE}.tgz" ]; then
    echo "Download from ${TBB_URL} to ${DST} failed"
    exit 1
  fi

  echo tar -xvf "${DST}/${TBB_PACKAGE}.tgz" -C "${DST}"
  tar -C "${DST}/${OS}" --strip-components=1 -xvf "${DST}/${TBB_PACKAGE}.tgz" "./${TBB_VERSION}"
  ls -al "${DST}/${OS}/"
  mv -f "${DST}/${OS}/${TBB_VERSION}/*" "${DST}/${OS}/"
  echo "Downloaded and unpacked oneTBB to ${DST}/${OS}"
else
  echo "oneTBB is already installed in ${DST}/${OS}"
fi
