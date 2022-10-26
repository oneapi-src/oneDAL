#!/bin/bash
#===============================================================================
# Copyright 2022 Intel Corporation
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

create_package() {
    # Args:
    # 1 - system
    # 2 - compiler type
    compiler_suffix=${2:-""}
    if [ "${compiler_suffix}" != "" ]; then
        compiler_suffix=_${compiler_suffix}
    fi

    if [ $1 = "lnx" ]; then
        platform=linux-x64
        tbb_platform=linux
        rls_postfix=lnx${compiler_suffix}/daal/latest
        lib_path=lib/intel64
        dl_prefix=lib
    elif [ $1 = "mac" ]; then
        platform=osx-x64
        tbb_platform=osx
        rls_postfix=mac${compiler_suffix}/daal/latest
        lib_path=lib
        dl_prefix=lib
    elif [ $1 = "win" ]; then
        platform=win-x64
        tbb_platform=win
        rls_postfix=win${compiler_suffix}/daal/latest
        lib_path=lib/intel64
        dl_prefix=""
    else
        echo "unknown platform"
        exit 1
    fi

    # version getting
    major_binary_version=$(cat __release_${rls_postfix}/include/services/library_version_info.h | grep __INTEL_DAAL_MAJOR_BINARY__ | head -n 1 | dos2unix | awk -F ' ' '{ print $3 }')
    minor_binary_version=$(cat __release_${rls_postfix}/include/services/library_version_info.h | grep __INTEL_DAAL_MINOR_BINARY__ | head -n 1 | dos2unix | awk -F ' ' '{ print $3 }')
    major_version=$(cat __release_${rls_postfix}/include/services/library_version_info.h | grep __INTEL_DAAL__ | head -n 1 | dos2unix | awk -F ' ' '{ print $3 }')
    minor_version=$(cat __release_${rls_postfix}/include/services/library_version_info.h | grep __INTEL_DAAL_MINOR__ | head -n 1 | dos2unix | awk -F ' ' '{ print $3 }')
    update_version=$(cat __release_${rls_postfix}/include/services/library_version_info.h | grep __INTEL_DAAL_UPDATE__ | head -n 1 | dos2unix | awk -F ' ' '{ print $3 }')
    dal_version=${major_version}.${minor_version}.${update_version}

    if [ $1 = "lnx" ]; then
        dl_postfix=.so.${major_binary_version}.${minor_binary_version}
    elif [ $1 = "mac" ]; then
        dl_postfix=.${major_binary_version}.${minor_binary_version}.dylib
    elif [ $1 = "win" ]; then
        dl_postfix=_dll.${major_binary_version}.lib
    fi

    pkg_name=inteldal.devel-reduced.${platform}.${dal_version}
    dal_root_prefix=${pkg_name}/build/native/daal
    tbb_root_prefix=${pkg_name}/build/native/tbb
    echo "Versions:"
    echo "MAJOR: ${major_version}, MINOR: ${minor_version}, UPDATE: ${update_version}, MAJOR BINARY: ${major_binary_version}"
    echo Creating $pkg_name
    mkdir -p ${dal_root_prefix}
    mkdir -p ${tbb_root_prefix}
    # nuspec generation
    sed "s/__VERSION__/${dal_version}/; s/__PLATFORM__/${platform}/; s/__TBB_PLATFORM__/${tbb_platform}/" deploy/nuget/devel-reduced/inteldal.devel-reduced.nuspec.tmpl > ${pkg_name}/inteldal.devel-reduced.${platform}.nuspec

    # ###### #
    # oneDAL #
    # ###### #

    # common part
    cp LICENSE ${pkg_name}
    # cmake
    cmake -DINSTALL_DIR=__release_${rls_postfix}/lib/cmake/oneDAL -P cmake/scripts/generate_config.cmake
    mkdir -p ${dal_root_prefix}/lib/cmake/oneDAL
    cp __release_${rls_postfix}/lib/cmake/oneDAL/* ${dal_root_prefix}/lib/cmake/oneDAL
    # env script
    cp -r __release_${rls_postfix}/env ${dal_root_prefix}
    # interfaces
    cp -r __release_${rls_postfix}/include ${dal_root_prefix}
    # dynamic libraries
    mkdir -p ${dal_root_prefix}/lib/intel64
    if [ $1 = "win" ]; then
        declare -a needed_libraries=(onedal_core)
    else
        declare -a needed_libraries=(onedal_core onedal_thread)
    fi
    for needed_library in "${needed_libraries[@]}"; do
        cp __release_${rls_postfix}/${lib_path}/${dl_prefix}${needed_library}${dl_postfix} ${dal_root_prefix}/lib/intel64
    done

    # win-x64 special part
    if [ $1 = "win" ]; then
        sed "s/__MAJOR_BINARY_VERSION__/${major_binary_version}/" deploy/nuget/devel-reduced/inteldal.devel-reduced.win-x64.targets > ${dal_root_prefix}/inteldal.devel-reduced.win-x64.targets
        cp deploy/nuget/devel-reduced/inteldal.devel-reduced.win-x64.xml ${dal_root_prefix}

        mkdir -p ${dal_root_prefix}/redist/intel64
        cp __release_${rls_postfix}/redist/intel64/onedal_core.${major_binary_version}.dll ${dal_root_prefix}/redist/intel64
        cp __release_${rls_postfix}/redist/intel64/onedal_thread.${major_binary_version}.dll ${dal_root_prefix}/redist/intel64

        cp __release_${rls_postfix}/lib/intel64/onedal_core_dll.${major_binary_version}.lib ${dal_root_prefix}/lib/intel64
    fi

    echo "oneDAL ${dal_version} is packed"

    # ###### #
    # oneTBB #
    # ###### #

    # download package
    tbb_version=2021.6.0
    tbb_download_prefix=https://github.com/oneapi-src/oneTBB/releases/download/v${tbb_version}
    if [ $1 = "lnx" ]; then
        tbb_package_name=oneapi-tbb-${tbb_version}-lin.tgz
        wget -q ${tbb_download_prefix}/${tbb_package_name}
        tar -xzf ${tbb_package_name}
        rm ${tbb_package_name}
    elif [ $1 = "mac" ]; then
        tbb_package_name=oneapi-tbb-${tbb_version}-mac.tgz
        wget -q ${tbb_download_prefix}/${tbb_package_name}
        tar -xzf ${tbb_package_name}
        rm ${tbb_package_name}
    elif [ $1 = "win" ]; then
        tbb_package_name=oneapi-tbb-${tbb_version}-win.zip
        wget -q ${tbb_download_prefix}/${tbb_package_name}
        unzip -q ${tbb_package_name}
        rm ${tbb_package_name}
    fi
    # interfaces
    cp -r oneapi-tbb-${tbb_version}/include ${tbb_root_prefix}
    # libraries
    cp -r oneapi-tbb-${tbb_version}/lib ${tbb_root_prefix}
    if [ $1 = "win" ]; then
        cp -r oneapi-tbb-${tbb_version}/redist ${tbb_root_prefix}
    fi
    # misc
    cp oneapi-tbb-${tbb_version}/LICENSE.txt ${tbb_root_prefix}
    cp oneapi-tbb-${tbb_version}/third-party-programs.txt ${tbb_root_prefix}

    rm -rf oneapi-tbb-${tbb_version}

    echo "oneTBB ${tbb_version} (dependency) is packed"

    cd ${pkg_name}; zip -q -9 -r ../${pkg_name}.nupkg *; cd ..
}

create_all_packages() {
    # Args:
    # 1 - compiler type
    create_package lnx $1
    create_package mac $1
    create_package win $1
}

set -eE

if [[ $# -eq 0 ]]; then
    create_all_packages $2
else
    create_package $1 $2
fi
