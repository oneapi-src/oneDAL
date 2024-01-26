# file: cmplr.clang.ref.arm.mk
#===============================================================================
# Copyright 2023-24 FUJITSU LIMITED
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
#  Clang definitions for makefile
#--

PLATs.clang = lnxarm

CMPLRDIRSUFF.clang = _clang

CORE.SERV.COMPILER.clang = generic

-Zl.clang =
-DEBC.clang = -g

COMPILER.lnx.clang= clang++ -march=armv8-a+sve \
                     -DDAAL_REF -DONEDAL_REF -DDAAL_CPU=sve -Werror -Wreturn-type
# Linker flags
link.dynamic.lnx.clang = clang++ -march=armv8-a+sve

pedantic.opts.clang = -pedantic \
                      -Wall \
                      -Wextra \
                      -Wno-unused-parameter

pedantic.opts.mac.clang = $(pedantic.opts.clang)
pedantic.opts.lnx.clang = $(pedantic.opts.clang)

# For SVE
a8sve_OPT.clang = $(-Q)march=armv8-a+sve
