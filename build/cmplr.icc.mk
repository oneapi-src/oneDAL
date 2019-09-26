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
#  Intel compiler defenitions for makefile
#--

PLATs.icc = lnx32e lnx32 win32e win32 mac32e

CMPLRDIRSUFF.icc =

CORE.SERV.COMPILER.icc = generic
-DEBC.icc = $(if $(OS_is_win),-debug:all -Z7,-g)

-Zl.icc = $(if $(OS_is_win),-Zl,) -mGLOB_freestanding=TRUE -mCG_no_libirc=TRUE
-Qopt = $(if $(OS_is_win),-Qopt-,-qopt-)

COMPILER.lnx.icc  = $(if $(COVFILE),cov01 -1; covc -i )icc -Werror -qopenmp-simd -Wreturn-type
COMPILER.lnx.icc += $(if $(COVFILE), $(if $(IA_is_ia32), $(-Q)m32, $(-Q)m64))
COMPILER.win.icc = icl -nologo -WX -Qopenmp-simd
COMPILER.mac.icc = icc -Werror -stdlib=libc++ -mmacosx-version-min=10.11 -Wreturn-type

# icc 16 does not support -qopenmp-simd option on macOS*
ifeq ($(if $(OS_is_mac),$(shell icc --version | grep "icc (ICC) 16"),),)
    COMPILER.mac.icc += -qopenmp-simd
endif

link.dynamic.lnx.icc = icc -no-cilk
link.dynamic.mac.icc = icc

daaldep.lnx32e.rt.icc = -static-intel
daaldep.lnx32.rt.icc  = -static-intel

p4_OPT.icc   = $(-Q)$(if $(OS_is_mac),xSSSE3,xSSE2)
mc_OPT.icc   = $(-Q)xSSSE3
mc3_OPT.icc  = $(-Q)xSSE4.2
avx_OPT.icc  = $(-Q)xAVX
avx2_OPT.icc = $(-Q)xCORE-AVX2
knl_OPT.icc  = $(if $(OS_is_mac),$(-Q)xCORE-AVX2,$(-Q)xMIC-AVX512)
skx_OPT.icc  = $(-Q)xCORE-AVX512 $(-Qopt)zmm-usage=high
#TODO add march opts in GCC style
