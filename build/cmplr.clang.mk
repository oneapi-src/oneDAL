# file: cmplt.clang.mk
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
#  Clang defenitions for makefile
#--

PLATs.clang = mac32e mac32

CMPLRDIRSUFF.clang = _clang

CORE.SERV.COMPILER.clang = generic

-Zl.clang = $(if $(OS_is_win),-Zl,)
-DEBC.clang = -g

COMPILER.lnx.clang = clang++ -D__int64="long long" -D__int32="int" $(if $(IA_is_ia32),-m32,-m64) -fgnu-runtime -Wno-inconsistent-missing-override -stdlib=libstdc++ -nostdinc++
COMPILER.mac.clang = clang++ -D__int64="long long" -D__int32="int" $(if $(IA_is_ia32),-m32,-m64) -fgnu-runtime -stdlib=libstdc++
#-Wno-inconsistent-missing-override

link.dynamic.lnx.clang = clang++ $(if $(IA_is_ia32),-m32,-m64)
link.dynamic.mac.clang = clang++ $(if $(IA_is_ia32),-m32,-m64)

p4_OPT.clang   = $(-Q)$(if $(OS_is_mac),$(if $(IA_is_ia32),march=nocona,march=core2),$(if $(IA_is_ia32),march=pentium4,march=nocona))
mc_OPT.clang   = $(-Q)$(if $(PLAT_is_mac32),march=nocona,march=core2)
mc3_OPT.clang  = $(-Q)march=corei7
avx_OPT.clang  = $(-Q)march=corei7-avx
avx2_OPT.clang = $(if $(OS_is_mac),$(-Q)march=corei7-avx,$(-Q)march=haswell)
knl_OPT.clang  = $(if $(OS_is_mac),$(-Q)march=corei7-avx,$(-Q)march=knl)
skx_OPT.clang  = $(if $(OS_is_mac),$(-Q)march=corei7-avx,$(-Q)march=skylake)
