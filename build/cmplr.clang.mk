# file: cmplt.clang.mk
#================================================== -*- makefile -*- vim:ft=make
# Copyright 2012-2019 Intel Corporation.
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

PLATs.clang = mac32e fbsd32e fbsd32

CMPLRDIRSUFF.clang = _clang

CORE.SERV.COMPILER.clang = generic

-Zl.clang =
-DEBC.clang = -g

COMPILER.mac.clang = clang++ -D__int64="long long" -D__int32="int" -m64 -fgnu-runtime -stdlib=libc++ -mmacosx-version-min=10.11
COMPILER.fbsd.clang = clang++ -D__int64="long long" -D__int32="int" $(if $(IA_is_ia32),-m32,-m64) -fgnu-runtime -Wno-inconsistent-missing-override -nostdinc++ -I/usr/include/c++/v1 -I/usr/local/include

link.dynamic.mac.clang = clang++ -m64
link.dynamic.fbsd.clang = clang++ $(if $(IA_is_ia32),-m32,-m64)

p4_OPT.clang   = $(-Q)march=nocona
mc_OPT.clang   = $(-Q)march=core2
mc3_OPT.clang  = $(-Q)march=nehalem
avx_OPT.clang  = $(-Q)march=sandybridge
avx2_OPT.clang = $(-Q)march=haswell
knl_OPT.clang  = $(-Q)march=knl
skx_OPT.clang  = $(-Q)march=skx
