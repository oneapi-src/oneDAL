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

prefix=${{pcfiledir}}/../../
exec_prefix=${{prefix}}
libdir=${{exec_prefix}}/{libdir}
includedir=${{prefix}}/include

#info
Name: oneDAL
Description: Intel(R) oneAPI Data Analytics Library
Version: [dal.version]
URL: https://software.intel.com/en-us/oneapi/onedal
#Link line
Libs: {libs}
#Compiler line
Cflags: {opts}
