# file: cmplt.dpcpp.mk
#===============================================================================
# Copyright 2012-2021 Intel Corporation
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
#  DPC++ Compiler defenitions for makefile
#--

PLATs.dpcpp = lnx32e win32e

CMPLRDIRSUFF.dpcpp = _dpcpp

CORE.SERV.COMPILER.dpcpp = generic

-Zl.dpcpp =
-DEBC.dpcpp = -g

COMPILER.lnx.dpcpp = dpcpp $(if $(IA_is_ia32),-m32,-m64) -stdlib=libstdc++ -fgnu-runtime -fwrapv \
                     -Werror -Wreturn-type
COMPILER.win.dpcpp = dpcpp $(if $(MSVC_RT_is_release),-MD, -MDd /debug:none) -nologo -WX -Wno-deprecated-declarations

link.dynamic.lnx.dpcpp = dpcpp $(if $(IA_is_ia32),-m32,-m64)
link.dynamic.win.dpcpp = dpcpp $(if $(IA_is_ia32),-m32,-m64)

pedantic.opts.lnx.dpcpp = -pedantic \
                          -Wall \
                          -Wextra \
                          -Wno-unused-parameter

p4_OPT.dpcpp   = -march=nocona
mc_OPT.dpcpp   = -march=core2
mc3_OPT.dpcpp  = -march=nehalem
avx_OPT.dpcpp  = -march=sandybridge
avx2_OPT.dpcpp = -march=haswell
knl_OPT.dpcpp  = -march=knl
skx_OPT.dpcpp  = -march=skx
