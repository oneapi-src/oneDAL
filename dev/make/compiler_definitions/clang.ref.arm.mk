# file: clang.ref.arm.mk
#===============================================================================
# Copyright contributors to the oneDAL project
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

include dev/make/compiler_definitions/clang.mk

PLATs.clang = lnxarm

COMPILER.lnx.clang.target = $(if $(filter yes,$(COMPILER_is_cross)),--target=aarch64-linux-gnu)

COMPILER.sysroot = $(if $(SYSROOT),--sysroot $(SYSROOT))

COMPILER.lnx.clang= clang++ -march=armv8-a+sve \
                     -DDAAL_REF -DONEDAL_REF -DDAAL_CPU=sve -Werror -Wreturn-type \
                     $(COMPILER.lnx.clang.target) \
                     $(COMPILER.sysroot)

# Linker flags
link.dynamic.lnx.clang = clang++ -march=armv8-a+sve \
                         $(COMPILER.lnx.clang.target) \
                         $(COMPILER.sysroot)

pedantic.opts.lnx.clang = $(pedantic.opts.clang)

# For SVE
a8sve_OPT.clang = $(-Q)march=armv8-a+sve
