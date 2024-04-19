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
#  g++ definitions for makefile
#  This file contains definitions common to gnu on a 32e (intel64) platform. It
#  should only be included from files which have more specializations (e.g.
#  gnu.mkl.32e.mk)
#--

include dev/make/compiler_definitions/gnu.mk

PLATs.gnu = lnx32e mac32e

COMPILER.all.gnu =  ${CXX} -m64 -fwrapv -fno-strict-overflow -fno-delete-null-pointer-checks \
                    -Werror -Wreturn-type

link.dynamic.all.gnu = ${CXX} -m64

pedantic.opts.lnx.gnu = $(pedantic.opts.all.gnu)
pedantic.opts.mac.gnu = $(pedantic.opts.all.gnu)

link.dynamic.lnx.gnu = $(link.dynamic.all.gnu)
link.dynamic.mac.gnu = $(link.dynamic.all.gnu)

p4_OPT.gnu   = $(-Q)march=nocona
mc3_OPT.gnu  = $(-Q)march=corei7
avx2_OPT.gnu = $(-Q)march=haswell
skx_OPT.gnu  = $(-Q)march=skylake
