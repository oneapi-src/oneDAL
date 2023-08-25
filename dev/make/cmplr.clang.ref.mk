# file: cmplt.clang.mk
#===============================================================================
# Copyright 2023 Intel Corporation
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

PLATs.clang = lnx32e mac32e

CMPLRDIRSUFF.clang = _clang

CORE.SERV.COMPILER.clang = generic

-Zl.clang =
-DEBC.clang = -g

COMPILER.mac.clang = clang++ -m64 -fgnu-runtime -stdlib=libc++ -mmacosx-version-min=10.15 -fwrapv \
                     -DDAAL_REF -Werror -Wreturn-type
COMPILER.lnx.clang = clang++ -m64 \
                     -DDAAL_REF -Werror -Wreturn-type

link.dynamic.mac.clang = clang++ -m64
link.dynamic.lnx.clang = clang++ -m64

pedantic.opts.clang = -pedantic \
                      -Wall \
                      -Wextra \
                      -Wno-unused-parameter

pedantic.opts.mac.clang = $(pedantic.opts.clang)
pedantic.opts.lnx.clang = $(pedantic.opts.clang)

p4_OPT.clang   = $(-Q)march=nocona
mc3_OPT.clang  = $(-Q)$(if $(OS_is_mac),march=nocona,march=nehalem) $(if $(OS_is_mac),$(-Q)mtune=nehalem)
avx2_OPT.clang = $(-Q)march=haswell
skx_OPT.clang  = $(-Q)march=skx
