#! /usr/bin/env python
#*******************************************************************************
# Copyright 2014-2017 Intel Corporation
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
#******************************************************************************/

# System imports
import os
import subprocess
import sys
from distutils.core import *
from distutils      import sysconfig
from setuptools     import setup, Extension
from os.path import join as jp
from distutils.sysconfig import get_config_vars

import numpy as np

npyver = int(np.__version__.split('.')[1])

if npyver == 9:
    print("Warning:  Detected numpy version {}".format(np.__version__))
    print("Numpy 1.10 or greater is strongly recommended.")
    print("Earlier versions have not been tested. Use at your own risk.")

if npyver < 9:
    sys.exit("Error: Detected numpy {}. The minimum requirement is 1.9, and >= 1.10 is strongly recommended".format(np.__version__))


daal_root = os.environ['DAALROOT']
daal_include = os.environ['DAAL_INCLUDE']
if not daal_include:
    daal_include = jp(daal_root, 'include')
cnc_root = os.environ['CNCROOT']
tbb_root = os.environ['TBBROOT']
IS_WIN = False
IS_MAC = False
IS_LIN = False

if 'linux' in sys.platform:
    IS_LIN = True
    lib_dir = jp('lib', 'intel64_lin')
elif sys.platform == 'darwin':
    IS_MAC = True
    lib_dir = 'lib'
elif sys.platform in ['win32', 'cygwin']:
    IS_WIN = True
    lib_dir = jp('lib', 'intel64_win')
else:
    assert False, sys.platform + ' not supported'

DAAL_DEFAULT_TYPE = 'double'

def get_sdl_cflags():
    if IS_LIN or IS_MAC:
        cflags = ['-fstack-protector', '-fPIC', '-D_DIST_', '-D_FORTIFY_SOURCE=2', '-Wformat', '-Wformat-security']
        if IS_LIN:
            return cflags
        elif IS_MAC:
            return cflags + []
    elif IS_WIN:
        return ['-GS']


def get_sdl_ldflags():
    if IS_LIN:
        return ['-Wl,-z,noexecstack', '-Wl,-z,relro', '-Wl,-z,now']
    elif IS_MAC:
        return []
    elif IS_WIN:
        return ['-NXCompat', '-DynamicBase']

def get_type_defines():
    daal_type_defines = ['DAAL_ALGORITHM_FP_TYPE', 'DAAL_SUMMARY_STATISTICS_TYPE', 'DAAL_DATA_TYPE']
    return ["-D{}={}".format(d, DAAL_DEFAULT_TYPE) for d in daal_type_defines]



def getpyexts():
    include_dir_plat = ['./daal4py/include', daal_include, cnc_root + '/include', tbb_root + '/include']
    using_intel = os.environ.get('cc', '') in ['icc', 'icpc', 'icl']
    eca = get_type_defines()
    ela = []

    if using_intel and IS_WIN:
        include_dir_plat.append(jp(os.environ.get('ICPP_COMPILER16', ''), 'compiler', 'include'))
        eca += ['-std=c++11', '-w']
    elif not using_intel and IS_WIN:
        eca += ['-wd4267', '-wd4244', '-wd4101', '-wd4996']
    else:
        eca += ['-std=c++11', '-w']

    # Security flags
    eca += get_sdl_cflags()
    ela += get_sdl_ldflags()

    if sys.version_info[0] >= 3:
        eca.append('-DSWIGPY_USE_CAPSULE_')

    if IS_WIN:
        libraries_plat = ['daal_thread', 'daal_core_dll', 'cnc']
    else:
        libraries_plat = ['daal_core', 'daal_thread', 'cnc']

    if IS_MAC:
        ela.append('-stdlib=libc++')
        ela.append("-Wl,-rpath,{}".format(jp(daal_root, lib_dir)))
        ela.append("-Wl,-rpath,{}".format(jp(cnc_root, 'lib', 'intel64')))
        ela.append("-Wl,-rpath,{}".format(jp(daal_root, '..', 'tbb', 'lib')))
    elif IS_WIN:
        ela.append('-IGNORE:4197')

    return [Extension('_daal4py',
                      ['./daal4py/daal_wrap.cpp'],
                      include_dirs=include_dir_plat + [np.get_include()],
                      extra_compile_args=eca,
                      extra_link_args=ela,
                      libraries=libraries_plat,
                      library_dirs=[jp(daal_root, lib_dir), jp(cnc_root, 'lib', 'intel64')],
                      language='c++')
        ]

cfg_vars = get_config_vars()
for key, value in get_config_vars().items():
    if type(value) == str:
        cfg_vars[key] = value.replace("-Wstrict-prototypes", "").replace('-DNDEBUG', '')

# daal setup
setup(  name        = "daal4py",
        description = "Higher Level Python API to Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL)",
        author      = "Intel",
        version     = "{{DAAL4PY_VERSION}}",
        classifiers=[
            'Development Status :: 4 - ALPHA',
            'Environment :: Console',
            'Intended Audience :: Developers',
            'Intended Audience :: System Administrators',
            'Intended Audience :: Other Audience',
            'Intended Audience :: Science/Research',
            'License :: Other/Proprietary License',
            'Operating System :: MacOS :: MacOS X',
            'Operating System :: Microsoft :: Windows',
            'Operating System :: POSIX :: Linux',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3',
            'Topic :: Scientific/Engineering',
            'Topic :: System',
            'Topic :: Software Development',
          ],
        setup_requires = ['numpy>=1.9'],
        packages = ['daal4py'],
        ext_modules = getpyexts(),
)
