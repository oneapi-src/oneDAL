#===============================================================================
# Copyright 2012-2016 Intel Corporation
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

PLATs.gnu = lnx32e lnx32

CMPLRDIRSUFF.gnu = _gnu

CORE.SERV.COMPILER.gnu = generic

-Zl.gnu = $(if $(OS_is_win),-Zl,)
-DEBC.gnu = -g

COMPILER.lnx.gnu = g++ -D__int64="long long" -D__int32="int" $(if $(IA_is_ia32),-m32,-m64)

link.dynamic.lnx.gnu = g++ $(if $(IA_is_ia32),-m32,-m64)

p4_OPT.gnu   = $(-Q)$(if $(OS_is_mac),$(if $(IA_is_ia32),march=nocona,march=core2),$(if $(IA_is_ia32),march=pentium4,march=nocona))
mc_OPT.gnu   = $(-Q)$(if $(PLAT_is_mac32),march=nocona,march=core2)
mc3_OPT.gnu  = $(-Q)march=corei7
avx_OPT.gnu  = $(-Q)march=sandybridge
avx2_OPT.gnu = $(-Q)march=haswell
knl_OPT.gnu  = $(-Q)march=haswell
skx_OPT.gnu  = $(-Q)march=haswell
