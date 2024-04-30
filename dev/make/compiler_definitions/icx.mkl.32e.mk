#===============================================================================
# Copyright 2022 Intel Corporation
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
#  Intel compiler definitions for makefile
#--

PLATs.icx = lnx32e win32e

CMPLRDIRSUFF.icx =

CORE.SERV.COMPILER.icx = generic

-Zl.icx = $(if $(OS_is_win),-Zl,) $(-Q)no-intel-lib=libirc
-DEBC.icx = -g

-Qopt = $(if $(OS_is_win),-qopt-,-qopt-)

COMPILER.lnx.icx = icx -m64 \
                     -Werror -Wreturn-type -qopenmp-simd -O3


COMPILER.win.icx = icx $(if $(MSVC_RT_is_release),-MD, -MDd) -O3 -WX -Qopenmp-simd -Wno-deprecated-declarations

link.dynamic.lnx.icx = icx -m64

pedantic.opts.icx = -pedantic \
                      -Wall \
                      -Wextra \
                      -Wwritable-strings \
                      -Wno-unused-parameter

pedantic.opts.icx_win = -Wall \
                      -Wextra \
                      -Wwritable-strings \
                      -Wno-unused-parameter

pedantic.opts.lnx.icx = $(pedantic.opts.icx)
pedantic.opts.win.icx = $(pedantic.opts.icx_win)

p4_OPT.icx   = -march=nocona
mc3_OPT.icx  = -march=nehalem
avx2_OPT.icx = -march=haswell
skx_OPT.icx  = -march=skx
