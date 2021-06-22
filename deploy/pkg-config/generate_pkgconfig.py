#===============================================================================
# Copyright 2021 Intel Corporation
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

##  Content:
##     Intel(R) oneDAL examples makefiles for windows generator
##******************************************************************************

import os
import sys
import glob
import argparse
from sys import platform

RESULT_PKG_CONFIGS = {
    'daal-static-seq-host': {
        'is_static': True,
        'is_threading': False,
        'dal_libs': ['onedal_core', 'onedal_sequential']
    }
}

if platform in ["linux2", "linux"]:
    PREF_LIB = "lib"
    SUFF_DYN_LIB = ".so"
    SUFF_STAT_LIB = ".a"
    TBB_LIBS = "-ltbb -ltbbmalloc"
    OTHER_LIBS = "-lpthread -ldl"
elif platform == "darwin":
    PREF_LIB = "lib"
    PREF_LIB = ".so"
    SUFF_DYN_LIB = ".so"
    TBB_LIBS = "-ltbb -ltbbmalloc"
    OTHER_LIBS = "-ldl"
elif platform in ["win32", "win64"]:
    PREF_LIB = ""
    SUFF_DYN_LIB = "dll.lib"
    SUFF_STAT_LIB = ".lib"
    TBB_LIBS = "tbb12.lib tbbmalloc.lib"
    OTHER_LIBS = " "
else:
    raise RuntimeError("Not support OS {}".format(platform))

def get_result_libs(is_static, is_threading, dal_libs):
    suffix = SUFF_STAT_LIB if is_static else SUFF_DYN_LIB
    out_lib = ["${{libdir}}{}{}{}".format(os.sep, PREF_LIB, lib, suffix) for lib in dal_libs]
    res_dal_libs = " ".join(out_lib)
    res_thread_libs = TBB_LIBS if is_threading else ''
    return res_dal_libs + ' ' + res_thread_libs + ' ' + OTHER_LIBS

def generate(config):
    with open(config.template_name, 'r') as pkg_template_file:
        pkg_template = pkg_template_file.read()

        for pkg_config in RESULT_PKG_CONFIGS:
            pack_pkg_config = RESULT_PKG_CONFIGS[pkg_config]
            libs = get_result_libs(**pack_pkg_config)
            libdir = 'intel64' + os.sep + 'lib'
            opts = '-I${includedir}' + os.sep + 'include'
            result_content = pkg_template.format(libdir=libdir, libs=libs, opts=opts)
            print(result_content)
            result_pkg_config_file = config.output_dir + os.sep + pkg_config + '.pc'
            with open(result_pkg_config_file, 'w') as f:
                f.write(result_content)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str,
                        help='Path to the output directory that will contain generated pkg-configs')
    parser.add_argument('--template_name', type=str, default='pkg-config.tpl',
                        help='Name of the solution template file')
    config = parser.parse_args()
    generate(config)

