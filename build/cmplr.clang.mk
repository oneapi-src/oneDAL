# file: cmplt.clang.mk
#===============================================================================
# Copyright 2012-2019 Intel Corporation
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

PLATs.clang = lnx32e lnx32 mac32e fbsd32e fbsd32

CMPLRDIRSUFF.clang = _clang

CORE.SERV.COMPILER.clang = generic

-Zl.clang =
-DEBC.clang = -g

COMPILER.mac.clang = clang++ -D__int64="long long" -D__int32="int" -m64 -fgnu-runtime -stdlib=libc++ -mmacosx-version-min=10.11 -fwrapv
COMPILER.fbsd.clang = clang++ -D__int64="long long" -D__int32="int" $(if $(IA_is_ia32),-m32,-m64) -fgnu-runtime -Wno-inconsistent-missing-override -nostdinc++ -I/usr/include/c++/v1 -I/usr/local/include
COMPILER.lnx.clang = clang++ -D__int64="long long" -D__int32="int" $(if $(IA_is_ia32),-m32,-m64)

link.dynamic.mac.clang = clang++ -m64
link.dynamic.fbsd.clang = clang++ $(if $(IA_is_ia32),-m32,-m64)
link.dynamic.lnx.clang = clang++ $(if $(IA_is_ia32),-m32,-m64)

p4_OPT.clang   = $(-Q)march=nocona
mc_OPT.clang   = $(-Q)march=core2
mc3_OPT.clang  = $(-Q)march=nehalem
avx_OPT.clang  = $(-Q)march=sandybridge
avx2_OPT.clang = $(-Q)march=haswell
knl_OPT.clang  = $(-Q)march=knl
skx_OPT.clang  = $(-Q)march=skx
