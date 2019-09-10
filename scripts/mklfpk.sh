#!/bin/bash
#===============================================================================
# Copyright 2018-2019 Intel Corporation
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



MKLFPK_URL_ROOT="https://github.com/intel/daal/releases/download/2019_u4/"
MKLFPK_VERSION="20180112_7"
MKLFPK_ARCH=32e

while [ 1 ] ; do
    if [ "$1" = "--help" ] ; then
        echo "Usage: $0 [32|32e]"
        echo "Usage example: $0 32e"
        exit 1
    elif [ "${1}" = "32" ] ; then
        MKLFPK_ARCH=ia32
    elif [ "${1}" = "32e" ] ; then
        MKLFPK_ARCH=intel64
    elif [ -z "$1" ] ; then
        break
    else
        echo "Error: unknown paramater $1!"
        echo "type $0 --help"
        exit 1
    fi
    shift
done

os=`uname`
if [ "$os" = "Linux" ]; then
  MKLFPK_OS=lnx
elif [ "$os" = "Darwin" ]; then
  MKLFPK_OS=mac
else
  echo "Cannot identify operating system. Try downloading package manually."
  exit 1
fi

MKLFPK_PACKAGE="mklfpk_${MKLFPK_OS}_${MKLFPK_VERSION}"
MKLFPK_URL=${MKLFPK_URL_ROOT}${MKLFPK_PACKAGE}.tgz
DST=`dirname $0`/../externals/mklfpk
mkdir -p ${DST}
DST=`cd ${DST};pwd`

if [ ! -e "${DST}/license.txt" ]; then
  if [ -x "$(command -v curl)" ]; then
    echo curl -L -o "${DST}/${MKLFPK_PACKAGE}.tgz" "${MKLFPK_URL}"
    curl -L -o "${DST}/${MKLFPK_PACKAGE}.tgz" "${MKLFPK_URL}"
  elif [ -x "$(command -v wget)" ]; then
    echo wget -O "${DST}/${MKLFPK_PACKAGE}.tgz" "${MKLFPK_URL}"
    wget -O "${DST}/${MKLFPK_PACKAGE}.tgz" "${MKLFPK_URL}"
  else
    echo "curl or wget not available"
    exit 1
  fi

  if [ \! $? ]; then
    echo "Download from ${MKLFPK_URL} to ${DST} failed"
    exit 1
  fi
  set -x 

  echo tar -xf "${DST}/${MKLFPK_PACKAGE}.tgz" -C $DST
  tar -xf "${DST}/${MKLFPK_PACKAGE}.tgz" -C $DST
  echo "Downloaded and unpacked Intel(R) MKL small libraries to $DST"
else
  echo "Intel(R) MKL small libraries are already installed in $DST"
fi