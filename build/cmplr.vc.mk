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
#  Visual Studio definitions for makefile
#--

PLATs.vc = win32e win32

CMPLRDIRSUFF.vc = _vc

CORE.SERV.COMPILER.vc = generic

-Zl.vc = -Zl
-DEBC.vc = -DEBUG -Z7

COMPILER.win.vc = cl -c -nologo -EHsc -WX

p4_OPT.vc   =
mc_OPT.vc   =
mc3_OPT.vc  =
avx_OPT.vc  = -arch:AVX
avx2_OPT.vc = -arch:AVX2
knl_OPT.vc  = -arch:AVX2
skx_OPT.vc  = -arch:AVX2
