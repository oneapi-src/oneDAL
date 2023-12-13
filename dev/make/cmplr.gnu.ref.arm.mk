#===============================================================================
# Copyright 2023 Intel Corporation
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
#  g++ defenitions for makefile
#--

PLATs.gnu = lnxarm

CMPLRDIRSUFF.gnu = _gnu

CORE.SERV.COMPILER.gnu = generic

-Zl.gnu =
-DEBC.gnu = -g

COMPILER.all.gnu =  ${CXX} -march=native -fwrapv -fno-strict-overflow -fno-delete-null-pointer-checks \
                    -DDAAL_REF -DONEDAL_REF -Werror -Wreturn-type

link.dynamic.all.gnu = ${CXX} -march=native

pedantic.opts.all.gnu = -pedantic \
                        -Wall \
                        -Wextra \
                        -Wno-unused-parameter

COMPILER.lnx.gnu = $(COMPILER.all.gnu)
link.dynamic.lnx.gnu = $(link.dynamic.all.gnu)
pedantic.opts.lnx.gnu = $(pedantic.opts.all.gnu)

COMPILER.mac.gnu = $(COMPILER.all.gnu)
link.dynamic.mac.gnu = $(link.dynamic.all.gnu)
pedantic.opts.mac.gnu = $(pedantic.opts.all.gnu)

a8sve_OPT.gnu = $(-Q)march=native
