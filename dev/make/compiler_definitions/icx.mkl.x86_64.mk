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

#++
#  Intel compiler definitions for makefile
#--

PLATs.icx = lnxx86_64 macx86_64

CMPLRDIRSUFF.icx = _icx

CORE.SERV.COMPILER.icx = generic

-Zl.icx =  -no-intel-lib=libirc
-DEBC.icx = -g

COMPILER.lnx.icx = icpx -m64 \
                     -Werror -Wreturn-type


link.dynamic.lnx.icx = icpx -m64

pedantic.opts.icx = -pedantic \
                      -Wall \
                      -Wextra \
                      -Wno-unused-parameter

pedantic.opts.lnx.icx = $(pedantic.opts.icx)

p4_OPT.icx   = $(-Q)march=nocona
mc3_OPT.icx  = $(-Q)march=nehalem
avx2_OPT.icx = $(-Q)march=haswell
skx_OPT.icx  = $(-Q)march=skx
