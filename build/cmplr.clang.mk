# file: cmplt.clang.mk
#================================================== -*- makefile -*- vim:ft=make
# Copyright 2012-2018 Intel Corporation.
#
# This software and the related documents are Intel copyrighted  materials,  and
# your use of  them is  governed by the  express license  under which  they were
# provided to you (License).  Unless the License provides otherwise, you may not
# use, modify, copy, publish, distribute,  disclose or transmit this software or
# the related documents without Intel's prior written permission.
#
# This software and the related documents  are provided as  is,  with no express
# or implied  warranties,  other  than those  that are  expressly stated  in the
# License.
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
COMPILER.mac.clang = clang++ -D__int64="long long" -D__int32="int" $(if $(IA_is_ia32),-m32,-m64) -fgnu-runtime -stdlib=libstdc++ -mmacosx-version-min=10.11


link.dynamic.lnx.clang = clang++ $(if $(IA_is_ia32),-m32,-m64)
link.dynamic.mac.clang = clang++ $(if $(IA_is_ia32),-m32,-m64)

p4_OPT.clang   = $(-Q)$(if $(OS_is_mac),$(if $(IA_is_ia32),march=nocona,march=core2),$(if $(IA_is_ia32),march=pentium4,march=nocona))
mc_OPT.clang   = $(-Q)$(if $(PLAT_is_mac32),march=nocona,march=core2)
mc3_OPT.clang  = $(-Q)march=corei7
avx_OPT.clang  = $(-Q)march=corei7-avx
avx2_OPT.clang = $(if $(OS_is_mac),$(-Q)march=corei7-avx,$(-Q)march=haswell)
knl_OPT.clang  = $(if $(OS_is_mac),$(-Q)march=corei7-avx,$(-Q)march=knl)
skx_OPT.clang  = $(if $(OS_is_mac),$(-Q)march=corei7-avx,$(-Q)march=skylake)
