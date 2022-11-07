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
    # 1 - template file path
    # 2 - nuspec release directory
    # 3 - release directory
    # 4 - platform
    # 5 - release version
    # 6 - major binary version
    # 7 - minor binary version
    # 8 - generation type [nuspec, full]
    # 9 - distribution type

    rls_dir=$3
    dal_version=$5
    major_binary_version=$6
    minor_binary_version=$7

    # platform
    if [ $4 = "lnx32e" ]; then
        platform=linux-x64
        rls_prefix=${rls_dir}/daal/latest
        dynamic_lib_path=lib/intel64
        static_lib_path=lib/intel64
        lib_prefix=libonedal
    elif [ $4 = "mac32e" ]; then
        platform=osx-x64
        rls_prefix=${rls_dir}/daal/latest
        dynamic_lib_path=lib
        static_lib_path=lib
        lib_prefix=libonedal
    elif [ $4 = "win32e" ]; then
        platform=win-x64
        rls_prefix=${rls_dir}/daal/latest
        dynamic_lib_path=redist/intel64
        static_lib_path=lib/intel64
        lib_prefix=onedal
    else
        echo "Unknown platform $4"
        exit 1
    fi

    # distribution type
    if [ $9 = "redist" ] || [ $9 = "devel" ] || [ $9 = "static" ]; then
        distr_type=$9
        if [ ${distr_type} = "redist" ]; then
            content="dynamic libraries and headers"
        elif [ ${distr_type} = "devel" ]; then
            content="dynamic and static libraries and headers"
        elif [ ${distr_type} = "static" ]; then
            content="static libraries and headers"
        fi
    else
        echo "Unknown distribution type $9"
        exit 1
    fi

    # nuspec generation
    sed_template="s/__DISTRTYPE__/${distr_type}/; s/__PLATFORM__/${platform}/; s/__VERSION__/${dal_version}/; s/__CONTENT__/${content}/; s/__YEAR__/$(date +%Y)/"
    sed "${sed_template}" $1 > $2/inteldal.${distr_type}.${platform}.nuspec

    if [ $8 = "full" ]; then
        # extension of libraries
        if [ $4 = "lnx32e" ]; then
            dl_postfix=.so.${major_binary_version}.${minor_binary_version}
            sl_postfix=.a
        elif [ $4 = "mac32e" ]; then
            dl_postfix=.${major_binary_version}.${minor_binary_version}.dylib
            sl_postfix=.a
        elif [ $4 = "win32e" ]; then
            dl_postfix=.${major_binary_version}.dll
            sl_postfix=.lib
        fi

        pkg_name=inteldal.${distr_type}.${platform}.${dal_version}
        nuget_pkgs_path=__nuget
        pkg_path=${nuget_pkgs_path}/${pkg_name}
        dal_root_prefix=${pkg_path}/build/native/daal
        echo "Creating ${pkg_name} at ${pkg_path}"
        mkdir -p ${dal_root_prefix}

        # nuspec
        cp $2/inteldal.${distr_type}.${platform}.nuspec ${pkg_path}

        # ###### #
        # oneDAL #
        # ###### #

        # common part
        cp LICENSE ${pkg_path}
        # -- cmake configs
        cmake -DINSTALL_DIR=${rls_prefix}/lib/cmake/oneDAL -P cmake/scripts/generate_config.cmake
        mkdir -p ${dal_root_prefix}/lib/cmake/oneDAL
        cp ${rls_prefix}/lib/cmake/oneDAL/* ${dal_root_prefix}/lib/cmake/oneDAL
        # -- env script
        cp -r ${rls_prefix}/env ${dal_root_prefix}
        # -- interfaces
        cp -r ${rls_prefix}/include ${dal_root_prefix}

        # dynamic libraries
        if [ ${distr_type} = "redist" ] || [ ${distr_type} = "devel" ]; then
            mkdir -p ${dal_root_prefix}/${dynamic_lib_path}
            cp ${rls_prefix}/${dynamic_lib_path}/${lib_prefix}*${dl_postfix} ${dal_root_prefix}/${dynamic_lib_path}
            # win-x64 special part
            if [ $1 = "win" ]; then
                mkdir -p ${dal_root_prefix}/${static_lib_path}
                cp ${rls_prefix}/${static_lib_path}/*_dll.${major_binary_version}.lib ${dal_root_prefix}/${static_lib_path}
            fi
        fi
        # static libraries
        if [ ${distr_type} = "static" ] || [ ${distr_type} = "devel" ]; then
            mkdir -p ${dal_root_prefix}/${static_lib_path}
            cp ${rls_prefix}/${static_lib_path}/${lib_prefix}*${sl_postfix} ${dal_root_prefix}/${static_lib_path}
        fi

        echo "oneDAL ${dal_version} is packed"

        # ###### #
        # oneTBB #
        # ###### #
        cp -r ${rls_dir}/tbb ${pkg_path}/build/native/

        echo "oneTBB (dependency) is packed"

        cd ${pkg_path}; zip -q -9 -r ../${pkg_name}.nupkg *; cd $OLDPWD
    fi
}

# Args:
# 1 - template file path
# 2 - nuspec release directory
# 3 - release directory
# 4 - platform
# 5 - release version
# 6 - major binary version
# 7 - minor binary version
# 8 - generation type [nuspec, full]
create_package $1 $2 $3 $4 $5 $6 $7 $8 redist
create_package $1 $2 $3 $4 $5 $6 $7 $8 static
create_package $1 $2 $3 $4 $5 $6 $7 $8 devel
