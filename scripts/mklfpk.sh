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


MKLFPK_URL_ROOT="https://github.com/intel/daal/releases/download/Dependencies/"
MKLFPK_VERSION="20180112_10"
MKLGPUFPK_VERSION="20191109"
WITH_GPU=true

while [ 1 ] ; do
    if [ "$1" = "--help" ] ; then
        echo "Usage: $0 [with_gpu=true|false]"
        echo "Usage example: $0 with_gpu=true"
        exit 1
    elif [ "${1:0:8}" = "with_gpu" ] ; then
        WITH_GPU=${1:9}
    elif [ -z "$1" ] ; then
        break
    else
        echo "Error: unknown paramater $1!"
        echo "type $0 --help"
        exit 1
    fi
    shift
done

function download_fpk()
{
  SRC=$1
  DST=$2
  FILENAME=$3

  mkdir -p ${DST}
  DST=`cd ${DST};pwd`

  if [ ! -e "${DST}/license.txt" ]; then
    if [ -x "$(command -v curl)" ]; then
      echo curl -L -o "${DST}/${FILENAME}" "${SRC}"
      curl -L -o "${DST}/${FILENAME}" "${SRC}"
    elif [ -x "$(command -v wget)" ]; then
      echo wget -O "${DST}/${FILENAME}" "${SRC}"
      wget -O "${DST}/${FILENAME}" "${SRC}"
    else
      echo "curl or wget not available"
      exit 1
    fi

    if [ $? -ne 0 -o ! -e "${DST}/${FILENAME}" ]; then
      echo "Download from ${SRC} to ${DST}/${FILENAME} failed"
      exit 1
    fi
    set -x

    echo tar -xf "${DST}/${FILENAME}" -C $DST
    tar -xf "${DST}/${FILENAME}" -C $DST
    echo "Downloaded and unpacked Intel(R) MKL small libraries to $DST"
  else
    echo "Intel(R) MKL small libraries are already installed in $DST"
  fi
}

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
MKLGPUFPK_PACKAGE="mklgpufpk_${MKLFPK_OS}_${MKLGPUFPK_VERSION}"
MKLFPK_URL=${MKLFPK_URL_ROOT}${MKLFPK_PACKAGE}.tgz
MKLGPUFPK_URL=${MKLFPK_URL_ROOT}${MKLGPUFPK_PACKAGE}.tgz
CPUDST=`dirname $0`/../externals/mklfpk
GPUDST=`dirname $0`/../externals/mklgpufpk

download_fpk "${MKLFPK_URL}" "${CPUDST}" "${MKLFPK_PACKAGE}.tgz"
if [ "${MKLFPK_OS}" != "mac" -a "${WITH_GPU}" == "true" ]; then
  download_fpk "${MKLGPUFPK_URL}" "${GPUDST}" "${MKLGPUFPK_PACKAGE}.tgz"
fi
