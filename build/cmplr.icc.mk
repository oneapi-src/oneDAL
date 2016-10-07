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
#  Intel compiler defenitions for makefile
#--

PLATs.icc = lnx32e lnx32 win32e win32 mac32 mac32e

CMPLRDIRSUFF.icc =

CORE.SERV.COMPILER.icc = generic
-DEBC.icc = $(if $(OS_is_win),-debug:all -Z7,-g)

-Zl.icc = $(if $(OS_is_win),-Zl,) -mGLOB_freestanding=TRUE -mCG_no_libirc=TRUE

COMPILER.lnx.icc = icc -Werror
                     # for "-Qdiag-disable:809" see Compiler CQ DPD200569375
COMPILER.win.icc = icl -nologo -Qdiag-disable:809 -WX
COMPILER.mac.icc = icc -Werror -stdlib=libstdc++

link.dynamic.lnx.icc = icc -no-cilk
link.dynamic.mac.icc = icc

daaldep.lnx32e.rt.icc = -static-intel
daaldep.lnx32.rt.icc  = -static-intel

p4_OPT.icc   = $(-Q)$(if $(OS_is_mac),$(if $(IA_is_ia32),xSSE3,xSSSE3),xSSE2)
mc_OPT.icc   = $(-Q)$(if $(PLAT_is_mac32),xSSE3,xSSSE3)
mc3_OPT.icc  = $(-Q)xSSE4.2
avx_OPT.icc  = $(-Q)xAVX
avx2_OPT.icc = $(-Q)xCORE-AVX2
knl_OPT.icc  = $(-Q)xMIC-AVX512
skx_OPT.icc  = $(-Q)xCORE-AVX512
#TODO add march opts in GCC style
