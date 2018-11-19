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
#  g++ defenitions for makefile
#--

PLATs.gnu = lnx32e lnx32

CMPLRDIRSUFF.gnu = _gnu

CORE.SERV.COMPILER.gnu = generic

-Zl.gnu =
-DEBC.gnu = -g

COMPILER.lnx.gnu = g++ -D__int64="long long" -D__int32="int" $(if $(IA_is_ia32),-m32,-m64)

link.dynamic.lnx.gnu = g++ $(if $(IA_is_ia32),-m32,-m64)

p4_OPT.gnu   = $(-Q)$(if $(IA_is_ia32),march=pentium4,march=nocona)
mc_OPT.gnu   = $(-Q)$(if $(IA_is_ia32),march=pentium4,march=nocona)
mc3_OPT.gnu  = $(-Q)march=corei7
avx_OPT.gnu  = $(-Q)march=sandybridge
avx2_OPT.gnu = $(-Q)march=haswell
knl_OPT.gnu  = $(-Q)march=haswell
skx_OPT.gnu  = $(-Q)march=haswell
