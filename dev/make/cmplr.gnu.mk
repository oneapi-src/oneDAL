#===============================================================================
# Copyright 2012 Intel Corporation
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
#  g++ defenitions for makefile
#--

PLATs.gnu = lnx32e mac32e

CMPLRDIRSUFF.gnu = _gnu

CORE.SERV.COMPILER.gnu = generic

-Zl.gnu =
-DEBC.gnu = -g

COMPILER.all.gnu =  ${CXX} $(if $(IA_is_ia32),-m32,-m64) -fwrapv -fno-strict-overflow -fno-delete-null-pointer-checks \
                    -Werror -Wreturn-type

link.dynamic.all.gnu = ${CXX} $(if $(IA_is_ia32),-m32,-m64)

pedantic.opts.all.gnu = -pedantic \
                        -Wall \
                        -Wextra \
                        -Wno-unused-parameter

COMPILER.lnx.gnu = $(COMPILER.all.gnu)
link.dynamic.lnx.gnu = $(link.dynamic.all.gnu) -Wl,-soname,libLIB_SUFFIX.so.LIB_MAJOR_VER
pedantic.opts.lnx.gnu = $(pedantic.opts.all.gnu)

COMPILER.mac.gnu = $(COMPILER.all.gnu)
link.dynamic.mac.gnu = $(link.dynamic.all.gnu)
pedantic.opts.mac.gnu = $(pedantic.opts.all.gnu)

p4_OPT.gnu   = $(-Q)$(if $(IA_is_ia32),march=pentium4,march=nocona)
mc_OPT.gnu   = $(-Q)$(if $(IA_is_ia32),march=pentium4,march=nocona)
mc3_OPT.gnu  = $(-Q)march=corei7
avx_OPT.gnu  = $(-Q)march=sandybridge
avx2_OPT.gnu = $(-Q)march=haswell
skx_OPT.gnu  = $(-Q)march=haswell
