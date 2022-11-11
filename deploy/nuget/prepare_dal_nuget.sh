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
    while [[ $# -gt 0 ]]; do
        key="$1"

        case $key in
            --template)
            template_path="$2"
            ;;
            --release-dir)
            rls_dir="$2"
            ;;
            --platform)
            platform="$2"
            ;;
            --ver)
            dal_version="$2"
            ;;
            --major-binary-ver)
            major_binary_version="$2"
            ;;
            --minor-binary-ver)
            minor_binary_version="$2"
            ;;
            --build-nupkg)
            build_nupkg="$2"
            ;;
            --distribution-type)
            distr_type="$2"
            ;;
            *)
            echo "Unknown option: $1"
            exit 1
            ;;
        esac
        shift
        shift
    done

    # platform specific
    if [ ${platform} = "lnx32e" ]; then
        platform=linux-x64
        rls_prefix=${rls_dir}/daal/latest
        dynamic_lib_path=lib/intel64
        static_lib_path=lib/intel64
        lib_prefix=libonedal
    elif [ ${platform} = "mac32e" ]; then
        platform=osx-x64
        rls_prefix=${rls_dir}/daal/latest
        dynamic_lib_path=lib
        static_lib_path=lib
        lib_prefix=libonedal
    elif [ ${platform} = "win32e" ]; then
        platform=win-x64
        rls_prefix=${rls_dir}/daal/latest
        dynamic_lib_path=redist/intel64
        static_lib_path=lib/intel64
        lib_prefix=onedal
    else
        echo "Unknown platform ${platform}"
        exit 1
    fi

    # distribution type
    if [ "${distr_type}" = "redist" ] || [ "${distr_type}" = "devel" ] || [ "${distr_type}" = "static" ]; then
        if [ ${distr_type} = "redist" ]; then
            content="dynamic libraries and headers"
        elif [ ${distr_type} = "devel" ]; then
            content="dynamic and static libraries and headers"
        elif [ ${distr_type} = "static" ]; then
            content="static libraries and headers"
        fi
    else
        echo "Unknown distribution type ${distr_type}"
        exit 1
    fi

    # nuspec generation
    sed_template="s/__DISTRTYPE__/${distr_type}/; s/__PLATFORM__/${platform}/; s/__VERSION__/${dal_version}/; s/__CONTENT__/${content}/; s/__YEAR__/$(date +%Y)/"
    sed "${sed_template}" ${template_path} > ${rls_dir}/daal/latest/nuspec/inteldal.${distr_type}.${platform}.nuspec

    if [ "${build_nupkg}" = "yes" ]; then
        # extension of libraries
        if [ "${platform}" = "linux-x64" ]; then
            dl_postfix=.so.${major_binary_version}.${minor_binary_version}
            sl_postfix=.a
        elif [ "${platform}" = "osx-x64" ]; then
            dl_postfix=.${major_binary_version}.${minor_binary_version}.dylib
            sl_postfix=.a
        elif [ "${platform}" = "win-x64" ]; then
            dl_postfix=.${major_binary_version}.dll
            sl_postfix=.lib
        fi

        pkg_name=inteldal.${distr_type}.${platform}.${dal_version}
        nuget_pkgs_path=__nuget
        pkg_path=${nuget_pkgs_path}/${pkg_name}
        dal_root_prefix=${pkg_path}/build/native/daal
        tbb_root_prefix=${pkg_path}/build/native/tbb
        echo "Creating ${pkg_name} at ${pkg_path}"
        mkdir -p ${dal_root_prefix}
        mkdir -p ${tbb_root_prefix}

        # oneDAL
        # -- license
        cp LICENSE ${pkg_path}
        # -- nuspec
        cp ${rls_dir}/daal/latest/nuspec/inteldal.${distr_type}.${platform}.nuspec ${pkg_path}
        # -- cmake configs
        cmake -DINSTALL_DIR=${rls_prefix}/lib/cmake/oneDAL -P cmake/scripts/generate_config.cmake
        mkdir -p ${dal_root_prefix}/lib/cmake/oneDAL
        cp ${rls_prefix}/lib/cmake/oneDAL/* ${dal_root_prefix}/lib/cmake/oneDAL
        # -- interfaces
        cp -r ${rls_prefix}/include ${dal_root_prefix}
        # -- dynamic libraries
        if [ ${distr_type} = "redist" ] || [ ${distr_type} = "devel" ]; then
            mkdir -p ${dal_root_prefix}/${dynamic_lib_path}
            cp ${rls_prefix}/${dynamic_lib_path}/${lib_prefix}*${dl_postfix} ${dal_root_prefix}/${dynamic_lib_path}
            # win-x64 special part
            if [ ${platform} = "win-x64" ]; then
                mkdir -p ${dal_root_prefix}/${static_lib_path}
                cp ${rls_prefix}/${static_lib_path}/*_dll.${major_binary_version}.lib ${dal_root_prefix}/${static_lib_path}
            fi
        fi
        # -- static libraries
        if [ ${distr_type} = "static" ] || [ ${distr_type} = "devel" ]; then
            mkdir -p ${dal_root_prefix}/${static_lib_path}
            cp ${rls_prefix}/${static_lib_path}/${lib_prefix}*${sl_postfix} ${dal_root_prefix}/${static_lib_path}
        fi

        echo "oneDAL ${dal_version} is packed"

        # oneTBB
        # -- interfaces
        cp -r ${rls_dir}/tbb/latest/include ${tbb_root_prefix}
        # -- cmake configs
        mkdir -p ${tbb_root_prefix}/lib
        cp -r ${rls_dir}/tbb/latest/lib/cmake ${pkg_path}/build/native/tbb/lib
        # -- libraries
        if [ ${platform} = "linux-x64" ]; then
            mkdir -p ${tbb_root_prefix}/lib/intel64/gcc4.8
            cp ${rls_dir}/tbb/latest/lib/intel64/* ${pkg_path}/build/native/tbb/lib/intel64/gcc4.8
        elif [ ${platform} = "osx-x64" ]; then
            cp ${rls_dir}/tbb/latest/lib/* ${pkg_path}/build/native/tbb/lib
        elif [ ${platform} = "win-x64" ]; then
            cp -r ${rls_dir}/tbb/latest/lib/intel64 ${pkg_path}/build/native/tbb/lib
        fi

        echo "oneTBB (dependency) is packed"

        # packaging
        cd ${pkg_path}; zip -q -9 -r ../${pkg_name}.nupkg *; cd $OLDPWD
    fi
}

create_package $@ --distribution-type redist
create_package $@ --distribution-type static
create_package $@ --distribution-type devel
