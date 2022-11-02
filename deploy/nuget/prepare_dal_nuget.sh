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
    # 1 - system [lnx, mac, win]
    # 2 - distribution type [redist, devel, static]
    # 3 - compiler type

    # paths to release dirs
    compiler_suffix=${3:-""}
    if [ "${compiler_suffix}" != "" ]; then
        compiler_suffix=_${compiler_suffix}
    fi

    if [ $1 = "lnx" ]; then
        platform=linux-x64
        rls_postfix=lnx${compiler_suffix}/daal/latest
        dynamic_lib_path=lib/intel64
        static_lib_path=lib/intel64
        lib_prefix=libonedal
    elif [ $1 = "mac" ]; then
        platform=osx-x64
        rls_postfix=mac${compiler_suffix}/daal/latest
        dynamic_lib_path=lib
        static_lib_path=lib
        lib_prefix=libonedal
    elif [ $1 = "win" ]; then
        platform=win-x64
        rls_postfix=win${compiler_suffix}/daal/latest
        dynamic_lib_path=redist/intel64
        static_lib_path=lib/intel64
        lib_prefix=onedal
    else
        echo "Unknown platform $1"
        exit 1
    fi

    # distribution type
    if [ $2 = "redist" ] || [ $2 = "devel" ] || [ $2 = "static" ]; then
    	distr_type=$2
    	if [ $2 = "redist" ]; then
    		content="dynamic libraries and headers"
    	elif [ $2 = "devel" ]; then
    		content="dynamic and static libraries and headers"
    	elif [ $2 = "static" ]; then
    		content="static libraries and headers"
    	fi
    else
    	echo "Unknown distribution type $2"
        exit 1
    fi

    # library version
    major_binary_version=$(cat __release_${rls_postfix}/include/services/library_version_info.h | grep __INTEL_DAAL_MAJOR_BINARY__ | head -n 1 | dos2unix | awk -F ' ' '{ print $3 }')
    minor_binary_version=$(cat __release_${rls_postfix}/include/services/library_version_info.h | grep __INTEL_DAAL_MINOR_BINARY__ | head -n 1 | dos2unix | awk -F ' ' '{ print $3 }')
    major_version=$(cat __release_${rls_postfix}/include/services/library_version_info.h | grep __INTEL_DAAL__ | head -n 1 | dos2unix | awk -F ' ' '{ print $3 }')
    minor_version=$(cat __release_${rls_postfix}/include/services/library_version_info.h | grep __INTEL_DAAL_MINOR__ | head -n 1 | dos2unix | awk -F ' ' '{ print $3 }')
    update_version=$(cat __release_${rls_postfix}/include/services/library_version_info.h | grep __INTEL_DAAL_UPDATE__ | head -n 1 | dos2unix | awk -F ' ' '{ print $3 }')
    dal_version=${major_version}.${minor_version}.${update_version}

    # get extension of libraries
    if [ $1 = "lnx" ]; then
        dl_postfix=.so.${major_binary_version}.${minor_binary_version}
        sl_postfix=.a
    elif [ $1 = "mac" ]; then
        dl_postfix=.${major_binary_version}.${minor_binary_version}.dylib
        sl_postfix=.a
    elif [ $1 = "win" ]; then
        dl_postfix=.${major_binary_version}.dll
        sl_postfix=.lib
    fi

    pkg_name=inteldal.${distr_type}.${platform}.${dal_version}
    dal_root_prefix=${pkg_name}/build/native/daal
    tbb_root_prefix=${pkg_name}/build/native/tbb
    echo "Versions:"
    echo "MAJOR: ${major_version}, MINOR: ${minor_version}, UPDATE: ${update_version}"
    echo "MAJOR BINARY: ${major_binary_version}, MINOR BINARY: ${minor_binary_version}"
    echo Creating $pkg_name
    mkdir -p ${dal_root_prefix}
    mkdir -p ${tbb_root_prefix}
    # nuspec generation
    sed_template="s/__DISTRTYPE__/${distr_type}/; s/__PLATFORM__/${platform}/; s/__VERSION__/${dal_version}/; s/__CONTENT__/${content}/; s/__YEAR__/$(date +%Y)/"
    sed "${sed_template}" deploy/nuget/inteldal.nuspec.template.txt > ${pkg_name}/inteldal.${distr_type}.${platform}.nuspec

    # ###### #
    # oneDAL #
    # ###### #

    # common part
    cp LICENSE ${pkg_name}
    # -- cmake configs
    cmake -DINSTALL_DIR=__release_${rls_postfix}/lib/cmake/oneDAL -P cmake/scripts/generate_config.cmake
    mkdir -p ${dal_root_prefix}/lib/cmake/oneDAL
    cp __release_${rls_postfix}/lib/cmake/oneDAL/* ${dal_root_prefix}/lib/cmake/oneDAL
    # -- env script
    cp -r __release_${rls_postfix}/env ${dal_root_prefix}
    # -- interfaces
    cp -r __release_${rls_postfix}/include ${dal_root_prefix}

    # dynamic libraries
    if [ ${distr_type} = "redist" ] || [ ${distr_type} = "devel" ]; then
        mkdir -p ${dal_root_prefix}/${dynamic_lib_path}
        cp __release_${rls_postfix}/${dynamic_lib_path}/${lib_prefix}*${dl_postfix} ${dal_root_prefix}/${dynamic_lib_path}
        # win-x64 special part
        if [ $1 = "win" ]; then
            mkdir -p ${dal_root_prefix}/${static_lib_path}
            cp __release_${rls_postfix}/${static_lib_path}/*_dll.${major_binary_version}.lib ${dal_root_prefix}/${static_lib_path}
        fi
    fi
    # static libraries
    if [ ${distr_type} = "static" ] || [ ${distr_type} = "devel" ]; then
        mkdir -p ${dal_root_prefix}/${static_lib_path}
        cp __release_${rls_postfix}/${static_lib_path}/${lib_prefix}*${sl_postfix} ${dal_root_prefix}/${static_lib_path}
    fi

    echo "oneDAL ${dal_version} is packed"

    # ###### #
    # oneTBB #
    # ###### #

    # download package
    tbb_version=2021.7.0
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
    # 1 - system
    # 2 - compiler type
    create_package $1 redist $2
    create_package $1 static $2
    create_package $1 devel $2
}

set -eE

# bash file arguments:
# 1 - system
# 2 - compiler (optional)
# 3 - distribution type (optional)
if [[ $# -eq 1 ]]; then
    system=$(uname)
    if [ ${system} = "Linux" ]; then
        create_all_packages lnx ""
    elif [ ${system} = "Darwin" ]; then
        create_all_packages mac ""
    else
        create_all_packages win ""
    fi
elif [[ $# -eq 1 ]]; then
    create_all_packages $1 ""
elif [[ $# -eq 2 ]]; then
    create_all_packages $1 $2
elif [[ $# -eq 3 ]]; then
    create_package $1 $2 $3
else
    echo "Wrong arguments"
    exit 1
fi
