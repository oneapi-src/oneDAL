#===============================================================================
# Copyright 2014-2021 Intel Corporation
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

COMPILERs = icc icx gnu clang vc
COMPILER ?= icc

$(if $(filter $(COMPILERs),$(COMPILER)),,$(error COMPILER must be one of $(COMPILERs)))

CPUs := sse2 ssse3 sse42 avx2 avx512_mic avx512 avx
CPUs.files := nrh mrm neh snb hsw skx knl
USERREQCPU := $(filter-out $(filter $(CPUs),$(REQCPU)),$(REQCPU))
USECPUS := $(if $(REQCPU),$(if $(USERREQCPU),$(error Unsupported value/s in REQCPU: $(USERREQCPU). List of supported CPUs: $(CPUs)),$(REQCPU)),$(CPUs))
USECPUS := $(if $(filter sse2,$(USECPUS)),$(USECPUS),sse2 $(USECPUS))

req-features = order-only second-expansion
ifneq ($(words $(req-features)),$(words $(filter $(req-features),$(.FEATURES))))
$(error This makefile requires a decent make, supporting $(req-features))
endif

.PHONY: help
help: ; $(info $(help))

#===============================================================================
# Common macros
#===============================================================================

ifeq (help,$(MAKECMDGOALS))
    PLAT:=win32e
endif

attr.lnx32e = lnx intel64 lin
attr.mac32e = mac intel64
attr.win32e = win intel64 win
attr.fbsd32e = fbsd intel64 fre

_OS := $(word 1,$(attr.$(PLAT)))
_IA := $(word 2,$(attr.$(PLAT)))
_OSc:= $(word 3,$(attr.$(PLAT)))

COMPILER_is_$(COMPILER)  := yes
OS_is_$(_OS)             := yes
IA_is_$(_IA)             := yes
PLAT_is_$(PLAT)          := yes

#===============================================================================
# Compiler specific part
#===============================================================================

ifeq ($(OS_is_lnx),yes)
GCC_TOOLCHAIN_PATH := $(realpath $(dir $(shell which gcc))/..)

ifeq ($(COMPILER_is_clang),yes)
C.COMPILE.gcc_toolchain := $(GCC_TOOLCHAIN_PATH)
endif

DPC.COMPILE.gcc_toolchain := $(GCC_TOOLCHAIN_PATH)
endif

include dev/make/cmplr.$(COMPILER).mk
include dev/make/cmplr.dpcpp.mk

$(if $(filter $(PLATs.$(COMPILER)),$(PLAT)),,$(error PLAT for $(COMPILER) must be defined to one of $(PLATs.$(COMPILER))))

#===============================================================================
# Dependencies generation
#===============================================================================

include dev/make/common.mk
include dev/make/deps.mk

#===============================================================================
# Common macros
#===============================================================================

AR_is_$(subst $(space),_,$(origin AR)) := yes

OSList          := lnx win mac fbsd

o      := $(if $(OS_is_win),obj,o)
a      := $(if $(OS_is_win),lib,a)
plib   := $(if $(OS_is_win),,lib)
scr    := $(if $(OS_is_win),bat,sh)
y      := $(notdir $(filter $(_OS)/%,lnx/so win/dll mac/dylib fbsd/so))
-Fo    := $(if $(OS_is_win),-Fo,-o)
-Q     := $(if $(OS_is_win),$(if $(COMPILER_is_vc),-,-Q),-)
-cxx11 := $(if $(COMPILER_is_vc),,$(-Q)std=c++11)
-cxx17 := $(if $(COMPILER_is_vc),/std:c++17,$(-Q)std=c++17)
-fPIC  := $(if $(OS_is_win),,-fPIC)
-Zl    := $(-Zl.$(COMPILER))
-DEBC  := $(if $(REQDBG),$(-DEBC.$(COMPILER)) -DDEBUG_ASSERT -DONEDAL_ENABLE_ASSERT) -DTBB_SUPPRESS_DEPRECATED_MESSAGES -D__TBB_LEGACY_MODE $(if $(REQPRF), -D__DAAL_ITTNOTIFY_ENABLE__)
-DEBJ  := $(if $(REQDBG),-g,-g:none)
-DEBL  := $(if $(REQDBG),$(if $(OS_is_win),-debug,))
-EHsc  := $(if $(OS_is_win),-EHsc,)
-isystem := $(if $(OS_is_win),-I,-isystem)
-sGRP  = $(if $(or $(OS_is_lnx),$(OS_is_fbsd)),-Wl$(comma)--start-group,)
-eGRP  = $(if $(or $(OS_is_lnx),$(OS_is_fbsd)),-Wl$(comma)--end-group,)
daalmake = $(if $(OS_is_fbsd),gmake,make)

p4_OPT   := $(p4_OPT.$(COMPILER))
mc_OPT   := $(mc_OPT.$(COMPILER))
mc3_OPT  := $(mc3_OPT.$(COMPILER))
avx_OPT  := $(avx_OPT.$(COMPILER))
avx2_OPT := $(avx2_OPT.$(COMPILER))
knl_OPT  := $(knl_OPT.$(COMPILER))
skx_OPT  := $(skx_OPT.$(COMPILER))

_OSr := $(if $(OS_is_win),win,$(if $(OS_is_lnx),lin,$(if $(OS_is_fbsd),fre,)))

USECPUS.files := $(subst sse2,nrh,$(subst ssse3,mrm,$(subst sse42,neh,$(subst avx,snb,$(subst avx2,hsw,$(subst avx512,skx,$(subst avx512_mic,knl,$(USECPUS))))))))
USECPUS.out := $(filter-out $(USECPUS),$(CPUs))
USECPUS.out.for.grep.filter := $(addprefix _,$(addsuffix _,$(subst $(space),_|_,$(USECPUS.out))))
USECPUS.out.grep.filter := $(if $(USECPUS.out),| grep -v -E '$(USECPUS.out.for.grep.filter)')
USECPUS.out.defs := $(subst sse2,^\#define DAAL_KERNEL_SSE2\b,$(subst ssse3,^\#define DAAL_KERNEL_SSSE3\b,\
                    $(subst sse42,^\#define DAAL_KERNEL_SSE42\b,$(subst avx,^\#define DAAL_KERNEL_AVX\b,\
                    $(subst avx2,^\#define DAAL_KERNEL_AVX2\b,$(subst avx512,^\#define DAAL_KERNEL_AVX512\b,\
                    $(subst avx512_mic,^\#define DAAL_KERNEL_AVX512_MIC\b,$(USECPUS.out))))))))
USECPUS.out.defs := $(subst $(space)^,|^,$(strip $(USECPUS.out.defs)))
USECPUS.out.defs.filter := $(if $(USECPUS.out.defs),sed $(sed.-b) $(sed.-i) -E -e 's/$(USECPUS.out.defs)/$(sed.eol)/')

#===============================================================================
# Paths
#===============================================================================

# LINUX release structure (under __release_lnx):
# daal
# daal/bin - platform independent binaries: env setters
# daal/examples - usage demonstrations
# daal/include - header files
# daal/lib - platform-independent libraries (jar files)
# daal/lib/intel64 - static and dynamic libraries for intel64

# macOS* release structure (under __release_mac):
# daal
# daal/bin - platform independent binaries: env setters
# daal/examples - usage demonstrations
# daal/include - header files
# daal/lib - platform-independent libraries (jar files), and Mach-O intel64 binaries

# WINDOWS release structure (under __release_win):
# daal
# daal/bin - platform independent binaries: env setters
# daal/examples - usage demonstrations
# daal/include - header files
# daal/lib - platform-independent libraries (jar files)
# daal/lib/intel64 - static and import libraries for intel64
# redist/intel64/daal - dlls for intel64

# FREEBSD release structure (under __release_fbsd):
# daal
# daal/bin - platform independent binaries: env setters
# daal/examples - usage demonstrations
# daal/include - header files
# daal/lib - platform-independent libraries (jar files)
# daal/lib/intel64 - static and dynamic libraries for intel64

# List of needed threadings layers can be specified in DAALTHRS.
# if DAALTHRS is empty, threading will be incapsulated to core
DAALTHRS ?= tbb seq
DAALAY   ?= a y

DIR:=.
CPPDIR:=$(DIR)/cpp
CPPDIR.daal:=$(CPPDIR)/daal
CPPDIR.onedal:=$(CPPDIR)/oneapi/dal
WORKDIR    ?= $(DIR)/__work$(CMPLRDIRSUFF.$(COMPILER))/$(PLAT)
RELEASEDIR ?= $(DIR)/__release_$(_OS)$(CMPLRDIRSUFF.$(COMPILER))
RELEASEDIR.daal        := $(RELEASEDIR)/daal/latest
RELEASEDIR.lib         := $(RELEASEDIR.daal)/lib
RELEASEDIR.env         := $(RELEASEDIR.daal)/env
RELEASEDIR.modulefiles := $(RELEASEDIR.daal)/modulefiles
RELEASEDIR.conf        := $(RELEASEDIR.daal)/config
RELEASEDIR.doc         := $(RELEASEDIR.daal)/documentation
RELEASEDIR.samples     := $(RELEASEDIR.daal)/samples
RELEASEDIR.jardir      := $(RELEASEDIR.daal)/lib
RELEASEDIR.libia       := $(RELEASEDIR.daal)/lib$(if $(OS_is_mac),,/$(_IA))
RELEASEDIR.include     := $(RELEASEDIR.daal)/include
RELEASEDIR.soia        := $(if $(OS_is_win),$(RELEASEDIR.daal)/redist/$(_IA),$(RELEASEDIR.libia))
WORKDIR.lib := $(WORKDIR)/daal/lib

COVFILE   := $(subst BullseyeStub,$(RELEASEDIR.daal)/Bullseye_$(_IA).cov,$(COVFILE))
COV.libia := $(if $(BULLSEYEROOT),$(BULLSEYEROOT)/lib)

MKLFPKDIR:= $(if $(wildcard $(DIR)/__deps/mklfpk/$(_OS)/*),$(DIR)/__deps/mklfpk,                            \
                $(if $(wildcard $(MKLFPKROOT)/include/*),$(subst \,/,$(MKLFPKROOT)),                              \
                    $(error Can`t find MKLFPK libs nether in $(DIR)/__deps/mklfpk/$(_OS) not in MKLFPKROOT.)))
MKLFPKDIR.include := $(MKLFPKDIR)/include $(MKLFPKDIR)/$(if $(OS_is_fbsd),lnx,$(_OS))/include
MKLFPKDIR.libia   := $(MKLFPKDIR)/$(if $(OS_is_fbsd),lnx,$(_OS))/lib/$(_IA)

topf = $(shell echo $1 | sed 's/ /111/g' | sed 's/(/222/g' | sed 's/)/333/g' | sed 's/\\/\//g')
frompf = $(shell echo $1 | sed 's/111/ /g' | sed 's/222/(/g' | sed 's/333/)/g')
frompf1 = $(shell echo $1 | sed 's/111/\\ /g' | sed 's/222/(/g' | sed 's/333/)/g')


#============================= TBB folders =====================================
TBBDIR := $(if $(wildcard $(DIR)/__deps/tbb/$(_OS)/*),$(DIR)/__deps/tbb/$(_OS)$(if $(OS_is_win),/tbb))
TBBDIR.2 := $(if $(TBBDIR),$(TBBDIR),$(call topf,$$TBBROOT))
TBBDIR.2 := $(if $(TBBDIR.2),$(TBBDIR.2),$(error Can`t find TBB neither in $(DIR)/__deps/tbb not in $$TBBROOT))

TBBDIR.include := $(if $(TBBDIR),$(TBBDIR)/include/tbb $(TBBDIR)/include)

TBBDIR.libia.prefix := $(TBBDIR.2)/lib

TBBDIR.libia.win.vc1  := $(if $(OS_is_win),$(if $(wildcard $(call frompf1,$(TBBDIR.libia.prefix))/$(_IA)/vc_mt),$(TBBDIR.libia.prefix)/$(_IA)/vc_mt,$(if $(wildcard $(call frompf1,$(TBBDIR.libia.prefix))/$(_IA)/vc14),$(TBBDIR.libia.prefix)/$(_IA)/vc14)))
TBBDIR.libia.win.vc2  := $(if $(OS_is_win),$(if $(TBBDIR.libia.win.vc1),,$(firstword $(filter $(call topf,$$TBBROOT)%,$(subst ;,$(space),$(call topf,$$LIB))))))
TBBDIR.libia.win.vc22 := $(if $(OS_is_win),$(if $(TBBDIR.libia.win.vc2),$(wildcard $(TBBDIR.libia.win.vc2)/tbb12.dll)))

TBBDIR.libia.win:= $(if $(OS_is_win),$(if $(TBBDIR.libia.win.vc22),$(TBBDIR.libia.win.vc2),$(if $(TBBDIR.libia.win.vc1),$(TBBDIR.libia.win.vc1),$(error Can`t find TBB libs nether in $(call frompf,$(TBBDIR.libia.prefix))/$(_IA)/vc_mt not in $(firstword $(filter $(TBBROOT)%,$(subst ;,$(space),$(LIB)))).))))

TBBDIR.libia.lnx.gcc1 := $(if $(OS_is_lnx),$(if $(wildcard $(TBBDIR.libia.prefix)/$(_IA)/gcc4.8/*),$(TBBDIR.libia.prefix)/$(_IA)/gcc4.8))
TBBDIR.libia.lnx.gcc2  := $(if $(OS_is_lnx),$(if $(TBBDIR.libia.lnx.gcc1),,$(firstword $(filter $(TBBROOT)%,$(subst :,$(space),$(LD_LIBRARY_PATH))))))
TBBDIR.libia.lnx.gcc22 := $(if $(OS_is_lnx),$(if $(TBBDIR.libia.lnx.gcc2),$(wildcard $(TBBDIR.libia.lnx.gcc2)/libtbb.so)))
TBBDIR.libia.lnx := $(if $(OS_is_lnx),$(if $(TBBDIR.libia.lnx.gcc22),$(TBBDIR.libia.lnx.gcc2),$(if $(TBBDIR.libia.lnx.gcc1),$(TBBDIR.libia.lnx.gcc1),$(error Can`t find TBB runtimes nether in $(TBBDIR.libia.prefix)/$(_IA)/gcc4.8 not in $(firstword $(filter $(TBBROOT)%,$(subst :,$(space),$(LD_LIBRARY_PATH)))).))))

TBBDIR.libia.mac.clang1  := $(if $(OS_is_mac),$(if $(wildcard $(TBBDIR.libia.prefix)/*),$(TBBDIR.libia.prefix)))
TBBDIR.libia.mac.clang2  := $(if $(OS_is_mac),$(if $(TBBDIR.libia.mac.clang1),,$(firstword $(filter $(TBBROOT)%,$(subst :,$(space),$(LIBRARY_PATH))))))
TBBDIR.libia.mac.clang22 := $(if $(OS_is_mac),$(if $(TBBDIR.libia.mac.clang2),$(wildcard $(TBBDIR.libia.mac.clang2)/libtbb.dylib)))
TBBDIR.libia.mac := $(if $(OS_is_mac),$(if $(TBBDIR.libia.mac.clang22),$(TBBDIR.libia.mac.clang2),$(if $(TBBDIR.libia.mac.clang1),$(TBBDIR.libia.mac.clang1),$(error Can`t find TBB runtimes nether in $(TBBDIR.libia.prefix) not in $(firstword $(filter $(TBBROOT)%,$(subst :,$(space),$(LIBRARY_PATH)))).))))

TBBDIR.libia.fbsd := $(if $(OS_is_fbsd),$(TBBDIR.libia.prefix))
TBBDIR.libia := $(TBBDIR.libia.$(_OS))

TBBDIR.soia.prefix := $(TBBDIR.2)/
TBBDIR.soia.prefix.1 := $(if $(OS_is_win),$(if $(wildcard $(call frompf1,$(TBBDIR.soia.prefix))redist/$(_IA)/vc_mt/*),$(TBBDIR.soia.prefix),$(TBBDIR.2)/../))

TBBDIR.soia.win  := $(if $(OS_is_win),$(if $(TBBDIR.libia.win.vc22),$(TBBDIR.libia.win.vc2),$(if $(wildcard $(call frompf1,$(TBBDIR.soia.prefix.1))redist/$(_IA)/vc_mt/*),$(TBBDIR.soia.prefix.1)redist/$(_IA)/vc_mt,$(if $(wildcard $(call frompf1,$(TBBDIR.soia.prefix.1))redist/$(_IA)/vc14/*),$(TBBDIR.soia.prefix.1)redist/$(_IA)/vc14,$(error Can`t find TBB runtimes nether in $(TBBDIR.soia.prefix.1)redist/$(_IA)/vc_mt not in $(firstword $(filter $(TBBROOT)%,$(subst ;,$(space),$(LIB)))).)))))
TBBDIR.soia.lnx  := $(if $(OS_is_lnx),$(TBBDIR.libia.lnx))
TBBDIR.soia.mac  := $(if $(OS_is_mac),$(TBBDIR.libia.mac))
TBBDIR.soia.fbsd := $(if $(OS_is_fbsd),$(TBBDIR.soia.prefix)/lib)
TBBDIR.soia := $(TBBDIR.soia.$(_OS))

RELEASEDIR.tbb       := $(RELEASEDIR)/tbb/latest
RELEASEDIR.tbb.libia := $(RELEASEDIR.tbb)/lib$(if $(OS_is_mac),,/$(_IA)$(if $(OS_is_win),/vc_mt,/$(TBBDIR.libia.lnx.gcc)))
RELEASEDIR.tbb.soia  := $(if $(OS_is_win),$(RELEASEDIR.tbb)/redist/$(_IA)/vc_mt,$(RELEASEDIR.tbb.libia))
releasetbb.LIBS_A := $(if $(OS_is_win),$(TBBDIR.libia)/tbb12.$(a) $(TBBDIR.libia)/tbbmalloc.$(a))
releasetbb.LIBS_Y := $(TBBDIR.soia)/$(plib)tbb$(if $(OS_is_win),12,).$(y) $(TBBDIR.soia)/$(plib)tbbmalloc.$(y)                                                           \
                     $(if $(or $(OS_is_lnx),$(OS_is_fbsd)), $(if $(wildcard $(TBBDIR.soia)/libtbbmalloc.so.2),$(wildcard $(TBBDIR.soia)/libtbbmalloc.so.2))\
                                                            $(if $(wildcard $(TBBDIR.soia)/libtbbmalloc.so.12),$(wildcard $(TBBDIR.soia)/libtbbmalloc.so.12))\
                                                            $(if $(wildcard $(TBBDIR.soia)/libtbb.so.2),$(wildcard $(TBBDIR.soia)/libtbb.so.2))\
                                                            $(if $(wildcard $(TBBDIR.soia)/libtbb.so.12),$(wildcard $(TBBDIR.soia)/libtbb.so.12))) \
                     $(if $(OS_is_mac),$(if $(wildcard $(TBBDIR.soia)/libtbb.12.dylib),$(wildcard $(TBBDIR.soia)/libtbb.12.dylib))\
                                       $(if $(wildcard $(TBBDIR.soia)/libtbbmalloc.2.dylib),$(wildcard $(TBBDIR.soia)/libtbbmalloc.2.dylib)))


RELEASEDIR.include.mklgpufpk := $(RELEASEDIR.include)/services/internal/sycl/math

MKLGPUFPKDIR:= $(if $(wildcard $(DIR)/__deps/mklgpufpk/$(_OS)/*),$(DIR)/__deps/mklgpufpk/$(_OS),$(subst \,/,$(MKLGPUFPKROOT)))
MKLGPUFPKDIR.include := $(MKLGPUFPKDIR)/include
MKLGPUFPKDIR.libia   := $(MKLGPUFPKDIR)/lib/$(_IA)

mklgpufpk.LIBS_A := $(MKLGPUFPKDIR.libia)/$(plib)daal_sycl.$(a)
mklgpufpk.HEADERS := $(MKLGPUFPKDIR.include)/mkl_dal_sycl.hpp $(MKLGPUFPKDIR.include)/mkl_dal_blas_sycl.hpp

#===============================================================================
# Release library names
#===============================================================================
include makefile.ver

y_full_name_postfix := $(if $(OS_is_win),,$(if $(OS_is_mac),.$(MAJORBINARY).$(MINORBINARY).$(y),.$(y).$(MAJORBINARY).$(MINORBINARY)))
y_major_name_postfix := $(if $(OS_is_win),,$(if $(OS_is_mac),.$(MAJORBINARY).$(y),.$(y).$(MAJORBINARY)))

core_a       := $(plib)onedal_core.$a
core_y       := $(plib)onedal_core$(if $(OS_is_win),.$(MAJORBINARY),).$y
oneapi_a     := $(plib)onedal.$a
oneapi_y     := $(plib)onedal$(if $(OS_is_win),.$(MAJORBINARY),).$y
oneapi_a.dpc := $(plib)onedal_dpc.$a
oneapi_y.dpc := $(plib)onedal_dpc$(if $(OS_is_win),.$(MAJORBINARY),).$y

thr_tbb_a := $(plib)onedal_thread.$a
thr_seq_a := $(plib)onedal_sequential.$a
thr_tbb_y := $(plib)onedal_thread$(if $(OS_is_win),.$(MAJORBINARY),).$y
thr_seq_y := $(plib)onedal_sequential$(if $(OS_is_win),.$(MAJORBINARY),).$y

daal_jar  := onedal.jar

jni_so    := $(plib)JavaAPI.$y

release.LIBS_A := $(core_a) \
                  $(if $(OS_is_win),$(foreach ilib,$(core_a),$(ilib:%.lib=%_dll.lib)),) \
                  $(if $(DAALTHRS),$(foreach i,$(DAALTHRS),$(thr_$(i)_a)),)
release.LIBS_Y := $(core_y) \
                  $(if $(DAALTHRS),$(foreach i,$(DAALTHRS),$(thr_$(i)_y)),)
release.LIBS_J := $(jni_so)
release.JARS = $(daal_jar)

release.ONEAPI.LIBS_A := $(oneapi_a) \
                         $(if $(OS_is_win),$(foreach ilib,$(oneapi_a),$(ilib:%.lib=%_dll.lib)),)
release.ONEAPI.LIBS_Y := $(oneapi_y)

release.ONEAPI.LIBS_A.dpc := $(oneapi_a.dpc) \
                             $(if $(OS_is_win),$(foreach ilib,$(oneapi_a.dpc),$(ilib:%.lib=%_dll.lib)),)
release.ONEAPI.LIBS_Y.dpc := $(oneapi_y.dpc)

# Libraries required for building
daaldep.lnx32e.mkl.thr := $(MKLFPKDIR.libia)/$(plib)daal_mkl_thread.$a
daaldep.lnx32e.mkl.seq := $(MKLFPKDIR.libia)/$(plib)daal_mkl_sequential.$a
daaldep.lnx32e.mkl := $(MKLFPKDIR.libia)/$(plib)daal_vmlipp_core.$a
daaldep.lnx32e.vml :=
daaldep.lnx32e.ipp := $(if $(COV.libia),$(COV.libia)/libcov.a)
daaldep.lnx32e.rt.thr := -L$(RELEASEDIR.tbb.soia) -ltbb -ltbbmalloc -lpthread $(daaldep.lnx32e.rt.$(COMPILER)) $(if $(COV.libia),$(COV.libia)/libcov.a)
daaldep.lnx32e.rt.seq := -lpthread $(daaldep.lnx32e.rt.$(COMPILER)) $(if $(COV.libia),$(COV.libia)/libcov.a)
daaldep.lnx32e.rt.dpc := -lpthread $(if $(COV.libia),$(COV.libia)/libcov.a)
daaldep.lnx32e.threxport := export_lnx32e.def

daaldep.lnx.threxport.create = grep -v -E '^(EXPORTS|;|$$)' $< $(USECPUS.out.grep.filter) | sed -e 's/^/-u /'

daaldep.win32e.mkl.thr := $(MKLFPKDIR.libia)/daal_mkl_thread.$a
daaldep.win32e.mkl.seq := $(MKLFPKDIR.libia)/daal_mkl_sequential.$a
daaldep.win32e.mkl := $(MKLFPKDIR.libia)/$(plib)daal_vmlipp_core.$a
daaldep.win32e.vml :=
daaldep.win32e.ipp :=
daaldep.win32e.rt.thr  := -LIBPATH:$(RELEASEDIR.tbb.libia) tbb12.lib tbbmalloc.lib libcpmt.lib libcmt.lib $(if $(CHECK_DLL_SIG),Wintrust.lib)
daaldep.win32e.rt.seq  := libcpmt.lib libcmt.lib $(if $(CHECK_DLL_SIG),Wintrust.lib)
daaldep.win32e.threxport := export.def

daaldep.win.threxport.create = grep -v -E '^(;|$$)' $< $(USECPUS.out.grep.filter)


daaldep.mac32e.mkl.thr := $(MKLFPKDIR.libia)/$(plib)daal_mkl_thread.$a
daaldep.mac32e.mkl.seq := $(MKLFPKDIR.libia)/$(plib)daal_mkl_sequential.$a
daaldep.mac32e.mkl := $(MKLFPKDIR.libia)/$(plib)daal_vmlipp_core.$a
daaldep.mac32e.vml :=
daaldep.mac32e.ipp :=
daaldep.mac32e.rt.thr := -L$(RELEASEDIR.tbb.soia) -ltbb -ltbbmalloc $(daaldep.mac32e.rt.$(COMPILER))
daaldep.mac32e.rt.seq := $(daaldep.mac32e.rt.$(COMPILER))
daaldep.mac32e.threxport := export_mac.def

daaldep.mac.threxport.create = grep -v -E '^(EXPORTS|;|$$)' $< $(USECPUS.out.grep.filter) | sed -e 's/^/-u /'


daaldep.fbsd32e.mkl.thr := $(MKLFPKDIR.libia)/$(plib)daal_mkl_thread.$a
daaldep.fbsd32e.mkl.seq := $(MKLFPKDIR.libia)/$(plib)daal_mkl_sequential.$a
daaldep.fbsd32e.mkl := $(MKLFPKDIR.libia)/$(plib)daal_vmlipp_core.$a
daaldep.fbsd32e.vml :=
daaldep.fbsd32e.ipp := $(if $(COV.libia),$(COV.libia)/libcov.a)
daaldep.fbsd32e.rt.thr := -L$(RELEASEDIR.tbb.soia) -ltbb -ltbbmalloc -lpthread $(daaldep.fbsd32e.rt.$(COMPILER)) $(if $(COV.libia),$(COV.libia)/libcov.a)
daaldep.fbsd32e.rt.seq := -lpthread $(daaldep.fbsd32e.rt.$(COMPILER)) $(if $(COV.libia),$(COV.libia)/libcov.a)
daaldep.fbsd32e.threxport := export_lnx32e.def

daaldep.fbsd.threxport.create = grep -v -E '^(EXPORTS|;|$$)' $< $(USECPUS.out.grep.filter) | sed -e 's/^/-Wl,-u -Wl,/'


daaldep.mkl.thr := $(daaldep.$(PLAT).mkl.thr)
daaldep.mkl.seq := $(daaldep.$(PLAT).mkl.seq)
daaldep.mkl     := $(daaldep.$(PLAT).mkl)
daaldep.vml     := $(daaldep.$(PLAT).vml)
daaldep.ipp     := $(daaldep.$(PLAT).ipp)
daaldep.rt.thr  := $(daaldep.$(PLAT).rt.thr)
daaldep.rt.seq  := $(daaldep.$(PLAT).rt.seq)
daaldep.rt.dpc  := $(daaldep.$(PLAT).rt.dpc)

# List oneAPI header files to populate release/include.
release.ONEAPI.HEADERS.exclude := ! -path "*/backend/*" ! -path "*.impl.*" ! -path "*_test.*" ! -path "*/test/*"
release.ONEAPI.HEADERS := $(shell find $(CPPDIR) -type f -name "*.hpp" $(release.ONEAPI.HEADERS.exclude))
release.ONEAPI.HEADERS.OSSPEC := $(foreach fn,$(release.ONEAPI.HEADERS),$(if $(filter %$(_OS),$(basename $(fn))),$(fn)))
release.ONEAPI.HEADERS.COMMON := $(foreach fn,$(release.ONEAPI.HEADERS),$(if $(filter $(addprefix %,$(OSList)),$(basename $(fn))),,$(fn)))
release.ONEAPI.HEADERS.COMMON := $(filter-out $(subst _$(_OS),,$(release.ONEAPI.HEADERS.OSSPEC)),$(release.ONEAPI.HEADERS.COMMON))

# List header files to populate release/include.
release.HEADERS := $(shell find $(CPPDIR.daal)/include -type f -name "*.h")
release.HEADERS.OSSPEC := $(foreach fn,$(release.HEADERS),$(if $(filter %$(_OS),$(basename $(fn))),$(fn)))
release.HEADERS.COMMON := $(foreach fn,$(release.HEADERS),$(if $(filter $(addprefix %,$(OSList)),$(basename $(fn))),,$(fn)))
release.HEADERS.COMMON := $(filter-out $(subst _$(_OS),,$(release.HEADERS.OSSPEC)),$(release.HEADERS.COMMON))

# List examples files to populate release/examples.
expat = %.java %.cpp %.h %.hpp %.txt %.csv %.cmake
expat += $(if $(OS_is_win),%.bat %.vcxproj %.filters %.user %.sln %makefile_$(_OS),%_$(_OS).lst %makefile_$(_OS) %_$(_OS).sh)
release.EXAMPLES.CPP   := $(filter $(expat),$(shell find examples/daal/cpp  -type f)) $(filter $(expat),$(shell find examples/daal/cpp_sycl -type f))
release.EXAMPLES.DATA  := $(filter $(expat),$(shell find examples/daal/data -type f))
release.EXAMPLES.JAVA  := $(filter $(expat),$(shell find examples/daal/java -type f))
release.ONEAPI.EXAMPLES.CPP  := $(filter $(expat),$(shell find examples/oneapi/cpp -type f))
release.ONEAPI.EXAMPLES.DPC  := $(filter $(expat),$(shell find examples/oneapi/dpc -type f))
release.ONEAPI.EXAMPLES.DATA := $(filter $(expat),$(shell find examples/oneapi/data -type f))

# List env files to populate release.
release.ENV = deploy/local/vars_$(_OS).$(scr)

# List modulefiles to populate release.
release.MODULEFILES = deploy/local/dal

# List config files to populate release.
release.CONF = deploy/local/config.txt

# List samples files to populate release/examples.
SAMPLES.srcdir:= $(DIR)/samples
spat = %.scala %.java %.cpp %.h %.txt %.csv %.html %.png %.parquet %.blob
spat += $(if $(OS_is_win),%.bat %.vcxproj %.filters %.user %.sln,%_$(_OS).lst %makefile_$(_OS) %.sh)
release.SAMPLES.CPP  := $(if $(wildcard $(SAMPLES.srcdir)/daal/cpp/*),                                                   \
                          $(if $(OS_is_mac),                                                                             \
                            $(filter $(spat),$(shell find $(SAMPLES.srcdir)/daal/cpp -not -wholename '*mpi*' -type f))   \
                          ,                                                                                              \
                            $(filter $(spat),$(shell find $(SAMPLES.srcdir)/daal/cpp -type f))                           \
                          )                                                                                              \
                        )
release.SAMPLES.JAVA := $(if $(wildcard $(SAMPLES.srcdir)/daal/java/*),                                                  \
                          $(if $(or $(OS_is_lnx),$(OS_is_mac),$(OS_is_fbsd)),                                            \
                            $(filter $(spat),$(shell find $(SAMPLES.srcdir)/daal/java -type f))                          \
                          )                                                                                              \
                        )
release.SAMPLES.SCALA := $(if $(wildcard $(SAMPLES.srcdir)/daal/scala/*),                                                \
                          $(if $(or $(OS_is_lnx),$(OS_is_mac),$(OS_is_fbsd)),                                            \
                            $(filter $(spat),$(shell find $(SAMPLES.srcdir)/daal/scala -type f))                         \
                          )                                                                                              \
                        )

# List doc files to populate release/documentation.
DOC.srcdir:= $(DIR)/../documentation
release.DOC := $(shell if [ -d $(DOC.srcdir) ]; then find $(DOC.srcdir) -not -wholename '*.svn*' -type f ;fi)
release.DOC.COMMON := $(foreach fn,$(release.DOC),$(if $(filter $(addprefix %,$(OSList)),$(basename $(fn))),,$(fn)))
release.DOC.OSSPEC := $(foreach fn,$(release.DOC),$(if $(filter %$(_OS),$(basename $(fn))),$(fn)))

#===============================================================================
# Core part
#===============================================================================
include makefile.lst

THR.srcdir       := $(CPPDIR.daal)/src/threading
CORE.srcdir      := $(CPPDIR.daal)/src/algorithms
EXTERNALS.srcdir := $(CPPDIR.daal)/src/externals

CORE.SERV.srcdir          := $(CPPDIR.daal)/src/services
CORE.SERV.COMPILER.srcdir := $(CPPDIR.daal)/src/services/compiler/$(CORE.SERV.COMPILER.$(COMPILER))

CORE.srcdirs  := $(CORE.SERV.srcdir) $(CORE.srcdir)                  \
                 $(if $(DAALTHRS),,$(THR.srcdir))                    \
                 $(addprefix $(CORE.SERV.srcdir)/, $(CORE.SERVICES)) \
                 $(addprefix $(CORE.srcdir)/, $(CORE.ALGORITHMS))    \
                 $(CORE.SERV.COMPILER.srcdir) $(EXTERNALS.srcdir)    \
                 $(CPPDIR.daal)/src/sycl \
                 $(CPPDIR.daal)/src/data_management

CORE.incdirs.common := $(RELEASEDIR.include) $(CPPDIR.daal) $(WORKDIR)
CORE.incdirs.thirdp := $(MKLFPKDIR.include) $(TBBDIR.include)
CORE.incdirs := $(CORE.incdirs.common) $(CORE.incdirs.thirdp)

containing = $(foreach v,$2,$(if $(findstring $1,$v),$v))
notcontaining = $(foreach v,$2,$(if $(findstring $1,$v),,$v))
cpy = cp -fp "$<" "$@"

CORE.tmpdir_a := $(WORKDIR)/core_static
CORE.tmpdir_y := $(WORKDIR)/core_dynamic
CORE.srcs     := $(notdir $(wildcard $(CORE.srcdirs:%=%/*.cpp)))
CORE.srcs     := $(if $(OS_is_mac),$(CORE.srcs),$(call notcontaining,_mac,$(CORE.srcs)))
CORE.objs_a   := $(CORE.srcs:%.cpp=$(CORE.tmpdir_a)/%.$o)
CORE.objs_a   := $(filter-out %core_threading_win_dll.$o,$(CORE.objs_a))
CORE.objs_y   := $(CORE.srcs:%.cpp=$(CORE.tmpdir_y)/%.$o)
CORE.objs_y   := $(if $(OS_is_win),$(CORE.objs_y),$(filter-out %core_threading_win_dll.$o,$(CORE.objs_y)))

CORE.objs_a_tmp := $(call containing,_fpt,$(CORE.objs_a))
CORE.objs_a     := $(call notcontaining,_fpt,$(CORE.objs_a))
CORE.objs_a_tpl := $(subst _fpt,_fpt_flt,$(CORE.objs_a_tmp)) $(subst _fpt,_fpt_dbl,$(CORE.objs_a_tmp))
CORE.objs_a     := $(CORE.objs_a) $(CORE.objs_a_tpl)

CORE.objs_a_tmp := $(call containing,_cpu,$(CORE.objs_a))
CORE.objs_a     := $(call notcontaining,_cpu,$(CORE.objs_a))
CORE.objs_a_tpl := $(foreach ccc,$(USECPUS.files),$(subst _cpu,_cpu_$(ccc),$(CORE.objs_a_tmp)))
CORE.objs_a     := $(CORE.objs_a) $(CORE.objs_a_tpl)

CORE.objs_y_tmp := $(call containing,_fpt,$(CORE.objs_y))
CORE.objs_y     := $(call notcontaining,_fpt,$(CORE.objs_y))
CORE.objs_y_tpl := $(subst _fpt,_fpt_flt,$(CORE.objs_y_tmp)) $(subst _fpt,_fpt_dbl,$(CORE.objs_y_tmp))
CORE.objs_y     := $(CORE.objs_y) $(CORE.objs_y_tpl)

CORE.objs_y_tmp := $(call containing,_cpu,$(CORE.objs_y))
CORE.objs_y     := $(call notcontaining,_cpu,$(CORE.objs_y))
CORE.objs_y_tpl := $(foreach ccc,$(USECPUS.files),$(subst _cpu,_cpu_$(ccc),$(CORE.objs_y_tmp)))
CORE.objs_y     := $(CORE.objs_y) $(CORE.objs_y_tpl)

-include $(CORE.tmpdir_a)/*.d
-include $(CORE.tmpdir_y)/*.d

$(CORE.tmpdir_a)/$(core_a:%.$a=%_link.txt): $(CORE.objs_a) | $(CORE.tmpdir_a)/. ; $(WRITE.PREREQS)
$(CORE.tmpdir_a)/$(core_a:%.$a=%_link.$a):  LOPT:=
$(CORE.tmpdir_a)/$(core_a:%.$a=%_link.$a):  $(CORE.tmpdir_a)/$(core_a:%.$a=%_link.txt) | $(CORE.tmpdir_a)/. ; $(LINK.STATIC)
$(WORKDIR.lib)/$(core_a):                   LOPT:=
$(WORKDIR.lib)/$(core_a):                   $(daaldep.ipp) $(daaldep.vml) $(daaldep.mkl) $(CORE.tmpdir_a)/$(core_a:%.$a=%_link.$a) ; $(LINK.STATIC)

$(WORKDIR.lib)/$(core_y): LOPT += $(-fPIC)
$(WORKDIR.lib)/$(core_y): LOPT += $(daaldep.rt.seq)
$(WORKDIR.lib)/$(core_y): LOPT += $(if $(OS_is_win),-IMPLIB:$(@:%.$(MAJORBINARY).dll=%_dll.lib),)
ifdef OS_is_win
$(WORKDIR.lib)/$(core_y:%.$(MAJORBINARY).dll=%_dll.lib): $(WORKDIR.lib)/$(core_y)
endif
$(CORE.tmpdir_y)/$(core_y:%.$y=%_link.txt): $(CORE.objs_y) $(if $(OS_is_win),$(CORE.tmpdir_y)/dll.res,) | $(CORE.tmpdir_y)/. ; $(WRITE.PREREQS)
$(WORKDIR.lib)/$(core_y):                   $(daaldep.ipp) $(daaldep.vml) $(daaldep.mkl) \
                                            $(if $(PLAT_is_win32e),$(CORE.srcdir)/export_win32e.def) \
                                            $(CORE.tmpdir_y)/$(core_y:%.$y=%_link.txt) ; $(LINK.DYNAMIC) ; $(LINK.DYNAMIC.POST)

$(CORE.objs_a): $(CORE.tmpdir_a)/inc_a_folders.txt
$(CORE.objs_a): COPT += $(-fPIC) $(-cxx11) $(-Zl) $(-DEBC)
$(CORE.objs_a): COPT += -D__TBB_NO_IMPLICIT_LINKAGE -DDAAL_NOTHROW_EXCEPTIONS \
                        -DDAAL_HIDE_DEPRECATED -DTBB_USE_ASSERT=0
$(CORE.objs_a): COPT += @$(CORE.tmpdir_a)/inc_a_folders.txt
$(filter %threading.$o, $(CORE.objs_a)): COPT += -D__DO_TBB_LAYER__
$(call containing,_nrh, $(CORE.objs_a)): COPT += $(p4_OPT)   -DDAAL_CPU=sse2
$(call containing,_mrm, $(CORE.objs_a)): COPT += $(mc_OPT)   -DDAAL_CPU=ssse3
$(call containing,_neh, $(CORE.objs_a)): COPT += $(mc3_OPT)  -DDAAL_CPU=sse42
$(call containing,_snb, $(CORE.objs_a)): COPT += $(avx_OPT)  -DDAAL_CPU=avx
$(call containing,_hsw, $(CORE.objs_a)): COPT += $(avx2_OPT) -DDAAL_CPU=avx2
$(call containing,_knl, $(CORE.objs_a)): COPT += $(knl_OPT)  -DDAAL_CPU=avx512_mic
$(call containing,_skx, $(CORE.objs_a)): COPT += $(skx_OPT)  -DDAAL_CPU=avx512
$(call containing,_flt, $(CORE.objs_a)): COPT += -DDAAL_FPTYPE=float
$(call containing,_dbl, $(CORE.objs_a)): COPT += -DDAAL_FPTYPE=double

$(CORE.objs_y): $(CORE.tmpdir_y)/inc_y_folders.txt
$(CORE.objs_y): COPT += $(-fPIC) $(-cxx11) $(-Zl) $(-DEBC)
$(CORE.objs_y): COPT += -D__DAAL_IMPLEMENTATION \
                        -D__TBB_NO_IMPLICIT_LINKAGE -DDAAL_NOTHROW_EXCEPTIONS \
                        -DDAAL_HIDE_DEPRECATED -DTBB_USE_ASSERT=0 \
                        $(if $(CHECK_DLL_SIG),-DDAAL_CHECK_DLL_SIG)
$(CORE.objs_y): COPT += @$(CORE.tmpdir_y)/inc_y_folders.txt
$(filter %threading.$o, $(CORE.objs_y)): COPT += -D__DO_TBB_LAYER__
$(call containing,_nrh, $(CORE.objs_y)): COPT += $(p4_OPT)   -DDAAL_CPU=sse2
$(call containing,_mrm, $(CORE.objs_y)): COPT += $(mc_OPT)   -DDAAL_CPU=ssse3
$(call containing,_neh, $(CORE.objs_y)): COPT += $(mc3_OPT)  -DDAAL_CPU=sse42
$(call containing,_snb, $(CORE.objs_y)): COPT += $(avx_OPT)  -DDAAL_CPU=avx
$(call containing,_hsw, $(CORE.objs_y)): COPT += $(avx2_OPT) -DDAAL_CPU=avx2
$(call containing,_knl, $(CORE.objs_y)): COPT += $(knl_OPT)  -DDAAL_CPU=avx512_mic
$(call containing,_skx, $(CORE.objs_y)): COPT += $(skx_OPT)  -DDAAL_CPU=avx512
$(call containing,_flt, $(CORE.objs_y)): COPT += -DDAAL_FPTYPE=float
$(call containing,_dbl, $(CORE.objs_y)): COPT += -DDAAL_FPTYPE=double

vpath
vpath %.cpp $(CORE.srcdirs)
vpath %.rc $(CORE.srcdirs)

$(CORE.tmpdir_a)/inc_a_folders.txt: makefile.lst | $(CORE.tmpdir_a)/. $(CORE.incdirs) ; $(call WRITE.PREREQS,$(addprefix -I, $(CORE.incdirs)),$(space))
$(CORE.tmpdir_y)/inc_y_folders.txt: makefile.lst | $(CORE.tmpdir_y)/. $(CORE.incdirs) ; $(call WRITE.PREREQS,$(addprefix -I, $(CORE.incdirs)),$(space))

$(CORE.tmpdir_a)/library_version_info.$(o): $(VERSION_DATA_FILE)
$(CORE.tmpdir_y)/library_version_info.$(o): $(VERSION_DATA_FILE)

define .compile.template.ay
$(eval template_source_cpp := $(subst .$o,.cpp,$(notdir $1)))
$(eval template_source_cpp := $(subst _fpt_flt,_fpt,$(template_source_cpp)))
$(eval template_source_cpp := $(subst _fpt_dbl,_fpt,$(template_source_cpp)))
$(eval template_source_cpp := $(subst _cpu_nrh,_cpu,$(template_source_cpp)))
$(eval template_source_cpp := $(subst _cpu_mrm,_cpu,$(template_source_cpp)))
$(eval template_source_cpp := $(subst _cpu_neh,_cpu,$(template_source_cpp)))
$(eval template_source_cpp := $(subst _cpu_snb,_cpu,$(template_source_cpp)))
$(eval template_source_cpp := $(subst _cpu_hsw,_cpu,$(template_source_cpp)))
$(eval template_source_cpp := $(subst _cpu_knl,_cpu,$(template_source_cpp)))
$(eval template_source_cpp := $(subst _cpu_skx,_cpu,$(template_source_cpp)))
$1: $(template_source_cpp) ; $(value C.COMPILE)
endef
$(foreach a,$(CORE.objs_a),$(eval $(call .compile.template.ay,$a,$(CORE.tmpdir_a))))
$(foreach a,$(CORE.objs_y),$(eval $(call .compile.template.ay,$a,$(CORE.tmpdir_y))))


$(CORE.tmpdir_y)/dll.res: $(VERSION_DATA_FILE)
$(CORE.tmpdir_y)/dll.res: RCOPT += $(addprefix -I, $(CORE.incdirs.common))
$(CORE.tmpdir_y)/%.res: %.rc | $(CORE.tmpdir_y)/. ; $(RC.COMPILE)


#===============================================================================
# oneAPI part
#===============================================================================
ONEAPI.tmpdir_a := $(WORKDIR)/oneapi_static
ONEAPI.tmpdir_y := $(WORKDIR)/oneapi_dynamic
ONEAPI.tmpdir_a.dpc := $(WORKDIR)/oneapi_dpc_static
ONEAPI.tmpdir_y.dpc := $(WORKDIR)/oneapi_dpc_dynamic

ONEAPI.incdirs.common := $(CPPDIR)
ONEAPI.incdirs.thirdp := $(CORE.incdirs.common) $(MKLFPKDIR.include) $(TBBDIR.include)
ONEAPI.incdirs := $(ONEAPI.incdirs.common) $(CORE.incdirs.thirdp)

ONEAPI.dispatcher_cpu = $(WORKDIR)/oneapi/dal/_dal_cpu_dispatcher_gen.hpp
ONEAPI.dispatcher_tag.nrh := -D__CPU_TAG__=oneapi::dal::backend::cpu_dispatch_default
ONEAPI.dispatcher_tag.mrm := -D__CPU_TAG__=oneapi::dal::backend::cpu_dispatch_ssse3
ONEAPI.dispatcher_tag.neh := -D__CPU_TAG__=oneapi::dal::backend::cpu_dispatch_sse42
ONEAPI.dispatcher_tag.snb := -D__CPU_TAG__=oneapi::dal::backend::cpu_dispatch_avx
ONEAPI.dispatcher_tag.knl := -D__CPU_TAG__=oneapi::dal::backend::cpu_dispatch_avx512_mic
ONEAPI.dispatcher_tag.hsw := -D__CPU_TAG__=oneapi::dal::backend::cpu_dispatch_avx2
ONEAPI.dispatcher_tag.skx := -D__CPU_TAG__=oneapi::dal::backend::cpu_dispatch_avx512

ONEAPI.srcdir := $(CPPDIR.onedal)
ONEAPI.srcdirs.base := $(ONEAPI.srcdir) \
                       $(ONEAPI.srcdir)/algo \
                       $(ONEAPI.srcdir)/table \
                       $(ONEAPI.srcdir)/graph \
                       $(ONEAPI.srcdir)/util \
                       $(ONEAPI.srcdir)/io \
                       $(addprefix $(ONEAPI.srcdir)/algo/, $(ONEAPI.ALGOS)) \
                       $(addprefix $(ONEAPI.srcdir)/io/, $(ONEAPI.IO))
ONEAPI.srcdirs.detail := $(foreach x,$(ONEAPI.srcdirs.base),$(shell find $x -maxdepth 1 -type d -name detail))
ONEAPI.srcdirs.backend := $(foreach x,$(ONEAPI.srcdirs.base),$(shell find $x -maxdepth 1 -type d -name backend))
ONEAPI.srcdirs := $(ONEAPI.srcdirs.base) $(ONEAPI.srcdirs.detail) $(ONEAPI.srcdirs.backend)

ONEAPI.srcs.all.exclude := ! -path "*_test.*" ! -path "*/test/*"
ONEAPI.srcs.all := $(foreach x,$(ONEAPI.srcdirs.base),$(shell find $x -maxdepth 1 -type f -name "*.cpp" $(ONEAPI.srcs.all.exclude))) \
                   $(foreach x,$(ONEAPI.srcdirs.detail),$(shell find $x -type f -name "*.cpp" $(ONEAPI.srcs.all.exclude))) \
                   $(foreach x,$(ONEAPI.srcdirs.backend),$(shell find $x -type f -name "*.cpp" $(ONEAPI.srcs.all.exclude)))
ONEAPI.srcs.all	:= $(ONEAPI.srcs.all:./%=%)
ONEAPI.srcs.dpc := $(filter %_dpc.cpp,$(ONEAPI.srcs.all))
ONEAPI.srcs     := $(filter-out %_dpc.cpp,$(ONEAPI.srcs.all))
ONEAPI.srcs.dpc := $(ONEAPI.srcs) $(ONEAPI.srcs.dpc)

ONEAPI.srcs.mangled     := $(subst /,-,$(ONEAPI.srcs))
ONEAPI.srcs.mangled.dpc := $(subst /,-,$(ONEAPI.srcs.dpc))

ONEAPI.objs_a     := $(ONEAPI.srcs.mangled:%.cpp=$(ONEAPI.tmpdir_a)/%.$o)
ONEAPI.objs_y     := $(ONEAPI.srcs.mangled:%.cpp=$(ONEAPI.tmpdir_y)/%.$o)
ONEAPI.objs_a.dpc := $(ONEAPI.srcs.mangled.dpc:%.cpp=$(ONEAPI.tmpdir_a.dpc)/%.$o)
ONEAPI.objs_y.dpc := $(ONEAPI.srcs.mangled.dpc:%.cpp=$(ONEAPI.tmpdir_y.dpc)/%.$o)
ONEAPI.objs_a.all := $(ONEAPI.objs_a) $(ONEAPI.objs_a.dpc)
ONEAPI.objs_y.all := $(ONEAPI.objs_y) $(ONEAPI.objs_y.dpc)

USECPUS.files_no_knl := $(filter-out knl,$(USECPUS.files))
# Populate _cpu files -> _cpu_%cpu_name%, where %cpu_name% is $(USECPUS.files)
# $1 Output variable name
# $2 List of object files
define .populate_cpus
$(eval non_cpu_files := $(call notcontaining,_cpu,$2))
$(eval cpu_files := $(call containing,_cpu,$2))
$(eval nrh_files := $(subst _nrh,_cpu_nrh,$(call containing,_nrh,$(non_cpu_files))))
$(eval mrm_files := $(subst _mrm,_cpu_mrm,$(call containing,_mrm,$(non_cpu_files))))
$(eval neh_files := $(subst _neh,_cpu_neh,$(call containing,_neh,$(non_cpu_files))))
$(eval snb_files := $(subst _snb,_cpu_snb,$(call containing,_snb,$(non_cpu_files))))
$(eval hsw_files := $(subst _hsw,_cpu_hsw,$(call containing,_hsw,$(non_cpu_files))))
$(eval skx_files := $(subst _skx,_cpu_skx,$(call containing,_skx,$(non_cpu_files))))
$(eval user_cpu_files := $(nrh_files) $(mrm_files) $(neh_files) $(snb_files) $(hsw_files) $(skx_files))
$(eval populated_cpu_files := $(foreach ccc,$(USECPUS.files_no_knl),$(subst _cpu,_cpu_$(ccc),$(cpu_files))))
$(eval populated_cpu_files := $(filter-out $(user_cpu_files),$(populated_cpu_files)))
$(eval $1 := $(non_cpu_files) $(populated_cpu_files))
endef

$(eval $(call .populate_cpus,ONEAPI.objs_a,$(ONEAPI.objs_a)))
$(eval $(call .populate_cpus,ONEAPI.objs_y,$(ONEAPI.objs_y)))
$(eval $(call .populate_cpus,ONEAPI.objs_a.dpc,$(ONEAPI.objs_a.dpc)))
$(eval $(call .populate_cpus,ONEAPI.objs_y.dpc,$(ONEAPI.objs_y.dpc)))

-include $(ONEAPI.tmpdir_a)/*.d
-include $(ONEAPI.tmpdir_y)/*.d
-include $(ONEAPI.tmpdir_a.dpc)/*.d
-include $(ONEAPI.tmpdir_y.dpc)/*.d

# Declares target for object file compilation
# $1: Object file
# $2: Temporary directory where object file is stored
# $3: Compiler id (C or DPC)
define .ONEAPI.compile
$(eval template_source_cpp := $(1:$2/%.$o=%.cpp))
$(eval template_source_cpp := $(subst -,/,$(template_source_cpp)))
$(eval template_source_cpp := $(subst _cpu_nrh,_cpu,$(template_source_cpp)))
$(eval template_source_cpp := $(subst _cpu_mrm,_cpu,$(template_source_cpp)))
$(eval template_source_cpp := $(subst _cpu_neh,_cpu,$(template_source_cpp)))
$(eval template_source_cpp := $(subst _cpu_snb,_cpu,$(template_source_cpp)))
$(eval template_source_cpp := $(subst _cpu_hsw,_cpu,$(template_source_cpp)))
$(eval template_source_cpp := $(subst _cpu_skx,_cpu,$(template_source_cpp)))
$1: $(template_source_cpp) | $(dir $1)/. ; $(value $3.COMPILE)
endef

# Declares target to compile static library
# $1: Path to the static library to be produced
# $2: List of dependencies
define .ONEAPI.declare_static_lib
$(1:%.$a=%_link.txt): $2 | $(dir $1)/. ; $(value WRITE.PREREQS)
$1: LOPT:=
$1: $(1:%.$a=%_link.txt) | $(dir $1)/. ; $(value LINK.STATIC)
endef

$(ONEAPI.dispatcher_cpu): | $(dir $(ONEAPI.dispatcher_cpu))/.
	$(if $(filter ssse3,$(USECPUS)),echo "#define ONEDAL_CPU_DISPATCH_SSSE3" >> $@)
	$(if $(filter sse42,$(USECPUS)),echo "#define ONEDAL_CPU_DISPATCH_SSE42" >> $@)
	$(if $(filter avx,$(USECPUS)),echo "#define ONEDAL_CPU_DISPATCH_AVX" >> $@)
	$(if $(filter avx2,$(USECPUS)),echo "#define ONEDAL_CPU_DISPATCH_AVX2" >> $@)
	$(if $(filter avx512,$(USECPUS)),echo "#define ONEDAL_CPU_DISPATCH_AVX512" >> $@)

# Create file with include paths
ONEAPI.include_options := $(addprefix -I, $(ONEAPI.incdirs.common)) \
                          $(addprefix $(-isystem), $(ONEAPI.incdirs.thirdp))

$(ONEAPI.tmpdir_a)/inc_a_folders.txt: | $(ONEAPI.tmpdir_a)/.
	$(call WRITE.PREREQS,$(ONEAPI.include_options),$(space))

$(ONEAPI.tmpdir_y)/inc_y_folders.txt: | $(ONEAPI.tmpdir_y)/.
	$(call WRITE.PREREQS,$(ONEAPI.include_options),$(space))

$(ONEAPI.tmpdir_a.dpc)/inc_a_folders.txt: | $(ONEAPI.tmpdir_a.dpc)/.
	$(call WRITE.PREREQS,$(ONEAPI.include_options),$(space))

$(ONEAPI.tmpdir_y.dpc)/inc_y_folders.txt: | $(ONEAPI.tmpdir_y.dpc)/.
	$(call WRITE.PREREQS,$(ONEAPI.include_options),$(space))

# Set compilation options to the object files which are part of STATIC lib
$(ONEAPI.objs_a): $(ONEAPI.dispatcher_cpu) $(ONEAPI.tmpdir_a)/inc_a_folders.txt
$(ONEAPI.objs_a): COPT += $(-fPIC) $(-cxx17) $(-Zl) $(-DEBC) $(-EHsc) $(pedantic.opts) \
                          -DDAAL_NOTHROW_EXCEPTIONS \
                          -DDAAL_HIDE_DEPRECATED \
                          -D__TBB_NO_IMPLICIT_LINKAGE \
                          -DTBB_USE_ASSERT=0 \
                           @$(ONEAPI.tmpdir_a)/inc_a_folders.txt
$(call containing,_nrh, $(ONEAPI.objs_a)): COPT += $(p4_OPT)   $(ONEAPI.dispatcher_tag.nrh)
$(call containing,_mrm, $(ONEAPI.objs_a)): COPT += $(mc_OPT)   $(ONEAPI.dispatcher_tag.mrm)
$(call containing,_neh, $(ONEAPI.objs_a)): COPT += $(mc3_OPT)  $(ONEAPI.dispatcher_tag.neh)
$(call containing,_snb, $(ONEAPI.objs_a)): COPT += $(avx_OPT)  $(ONEAPI.dispatcher_tag.snb)
$(call containing,_hsw, $(ONEAPI.objs_a)): COPT += $(avx2_OPT) $(ONEAPI.dispatcher_tag.hsw)
$(call containing,_knl, $(ONEAPI.objs_a)): COPT += $(avx2_OPT) $(ONEAPI.dispatcher_tag.knl)
$(call containing,_skx, $(ONEAPI.objs_a)): COPT += $(skx_OPT)  $(ONEAPI.dispatcher_tag.skx)

$(ONEAPI.objs_a.dpc): $(ONEAPI.dispatcher_cpu) $(ONEAPI.tmpdir_a.dpc)/inc_a_folders.txt
$(ONEAPI.objs_a.dpc): COPT += $(-fPIC) $(-cxx17) $(-DEBC) $(-EHsc) $(pedantic.opts.dpcpp) \
                              -DDAAL_NOTHROW_EXCEPTIONS \
                              -DDAAL_HIDE_DEPRECATED \
                              -DDAAL_SYCL_INTERFACE \
                              -DONEDAL_DATA_PARALLEL \
                              -D__TBB_NO_IMPLICIT_LINKAGE \
                              -DTBB_USE_ASSERT=0 \
                               @$(ONEAPI.tmpdir_a.dpc)/inc_a_folders.txt
$(call containing,_nrh, $(ONEAPI.objs_a.dpc)): COPT += $(p4_OPT.dpcpp)   $(ONEAPI.dispatcher_tag.nrh)
$(call containing,_mrm, $(ONEAPI.objs_a.dpc)): COPT += $(mc_OPT.dpcpp)   $(ONEAPI.dispatcher_tag.mrm)
$(call containing,_neh, $(ONEAPI.objs_a.dpc)): COPT += $(mc3_OPT.dpcpp)  $(ONEAPI.dispatcher_tag.neh)
$(call containing,_snb, $(ONEAPI.objs_a.dpc)): COPT += $(avx_OPT.dpcpp)  $(ONEAPI.dispatcher_tag.snb)
$(call containing,_hsw, $(ONEAPI.objs_a.dpc)): COPT += $(avx2_OPT.dpcpp) $(ONEAPI.dispatcher_tag.hsw)
$(call containing,_knl, $(ONEAPI.objs_a.dpc)): COPT += $(avx2_OPT.dpcpp) $(ONEAPI.dispatcher_tag.knl)
$(call containing,_skx, $(ONEAPI.objs_a.dpc)): COPT += $(skx_OPT.dpcpp)  $(ONEAPI.dispatcher_tag.skx)

# Set compilation options to the object files which are part of DYNAMIC lib
$(ONEAPI.objs_y): $(ONEAPI.dispatcher_cpu) $(ONEAPI.tmpdir_y)/inc_y_folders.txt
$(ONEAPI.objs_y): COPT += $(-fPIC) $(-cxx17) $(-Zl) $(-DEBC) $(-EHsc) $(pedantic.opts) \
                          -DDAAL_NOTHROW_EXCEPTIONS \
                          -DDAAL_HIDE_DEPRECATED \
                          $(if $(CHECK_DLL_SIG),-DDAAL_CHECK_DLL_SIG) \
                          -D__ONEDAL_ENABLE_DLL_EXPORT__ \
                          -D__TBB_NO_IMPLICIT_LINKAGE \
                          -DTBB_USE_ASSERT=0 \
                          @$(ONEAPI.tmpdir_y)/inc_y_folders.txt
$(call containing,_nrh, $(ONEAPI.objs_y)): COPT += $(p4_OPT)   $(ONEAPI.dispatcher_tag.nrh)
$(call containing,_mrm, $(ONEAPI.objs_y)): COPT += $(mc_OPT)   $(ONEAPI.dispatcher_tag.mrm)
$(call containing,_neh, $(ONEAPI.objs_y)): COPT += $(mc3_OPT)  $(ONEAPI.dispatcher_tag.neh)
$(call containing,_snb, $(ONEAPI.objs_y)): COPT += $(avx_OPT)  $(ONEAPI.dispatcher_tag.snb)
$(call containing,_hsw, $(ONEAPI.objs_y)): COPT += $(avx2_OPT) $(ONEAPI.dispatcher_tag.hsw)
$(call containing,_knl, $(ONEAPI.objs_y)): COPT += $(avx2_OPT) $(ONEAPI.dispatcher_tag.knl)
$(call containing,_skx, $(ONEAPI.objs_y)): COPT += $(skx_OPT)  $(ONEAPI.dispatcher_tag.skx)

$(ONEAPI.objs_y.dpc): $(ONEAPI.dispatcher_cpu) $(ONEAPI.tmpdir_y.dpc)/inc_y_folders.txt
$(ONEAPI.objs_y.dpc): COPT += $(-fPIC) $(-cxx17) $(-DEBC) $(-EHsc) $(pedantic.opts.dpcpp) \
                              -DDAAL_NOTHROW_EXCEPTIONS \
                              -DDAAL_HIDE_DEPRECATED \
                              -DDAAL_SYCL_INTERFACE \
                              -DONEDAL_DATA_PARALLEL \
                              $(if $(CHECK_DLL_SIG),-DDAAL_CHECK_DLL_SIG) \
                              -D__ONEDAL_ENABLE_DLL_EXPORT__ \
                              -D__TBB_NO_IMPLICIT_LINKAGE \
                              -DTBB_USE_ASSERT=0 \
                              @$(ONEAPI.tmpdir_y.dpc)/inc_y_folders.txt
$(call containing,_nrh, $(ONEAPI.objs_y.dpc)): COPT += $(p4_OPT.dpcpp)   $(ONEAPI.dispatcher_tag.nrh)
$(call containing,_mrm, $(ONEAPI.objs_y.dpc)): COPT += $(mc_OPT.dpcpp)   $(ONEAPI.dispatcher_tag.mrm)
$(call containing,_neh, $(ONEAPI.objs_y.dpc)): COPT += $(mc3_OPT.dpcpp)  $(ONEAPI.dispatcher_tag.neh)
$(call containing,_snb, $(ONEAPI.objs_y.dpc)): COPT += $(avx_OPT.dpcpp)  $(ONEAPI.dispatcher_tag.snb)
$(call containing,_hsw, $(ONEAPI.objs_y.dpc)): COPT += $(avx2_OPT.dpcpp) $(ONEAPI.dispatcher_tag.hsw)
$(call containing,_knl, $(ONEAPI.objs_y.dpc)): COPT += $(avx2_OPT.dpcpp) $(ONEAPI.dispatcher_tag.knl)
$(call containing,_skx, $(ONEAPI.objs_y.dpc)): COPT += $(skx_OPT.dpcpp)  $(ONEAPI.dispatcher_tag.skx)

$(foreach x,$(ONEAPI.objs_a),$(eval $(call .ONEAPI.compile,$x,$(ONEAPI.tmpdir_a),C)))
$(foreach x,$(ONEAPI.objs_y),$(eval $(call .ONEAPI.compile,$x,$(ONEAPI.tmpdir_y),C)))
$(foreach x,$(ONEAPI.objs_a.dpc),$(eval $(call .ONEAPI.compile,$x,$(ONEAPI.tmpdir_a.dpc),DPC)))
$(foreach x,$(ONEAPI.objs_y.dpc),$(eval $(call .ONEAPI.compile,$x,$(ONEAPI.tmpdir_y.dpc),DPC)))

# Create Host and DPC++ oneapi libraries
$(eval $(call .ONEAPI.declare_static_lib,$(WORKDIR.lib)/$(oneapi_a),$(ONEAPI.objs_a)))
$(eval $(call .ONEAPI.declare_static_lib,$(WORKDIR.lib)/$(oneapi_a.dpc),$(ONEAPI.objs_a.dpc)))

$(ONEAPI.tmpdir_y)/$(oneapi_y:%.$y=%_link.txt): \
    $(ONEAPI.objs_y) $(if $(OS_is_win),$(ONEAPI.tmpdir_y)/dll.res,) | $(ONEAPI.tmpdir_y)/. ; $(WRITE.PREREQS)
$(WORKDIR.lib)/$(oneapi_y): \
    $(daaldep.ipp) $(daaldep.vml) $(daaldep.mkl) \
    $(ONEAPI.tmpdir_y)/$(oneapi_y:%.$y=%_link.txt) ; $(LINK.DYNAMIC) ; $(LINK.DYNAMIC.POST)
$(WORKDIR.lib)/$(oneapi_y): LOPT += $(-fPIC)
$(WORKDIR.lib)/$(oneapi_y): LOPT += $(daaldep.rt.seq)
$(WORKDIR.lib)/$(oneapi_y): LOPT += $(if $(OS_is_win),-IMPLIB:$(@:%.$(MAJORBINARY).dll=%_dll.lib),)
$(WORKDIR.lib)/$(oneapi_y): LOPT += $(if $(OS_is_win),$(WORKDIR.lib)/$(core_y:%.$(MAJORBINARY).dll=%_dll.lib))
ifdef OS_is_win
$(WORKDIR.lib)/$(oneapi_y:%.$(MAJORBINARY).dll=%_dll.lib): $(WORKDIR.lib)/$(oneapi_y)
endif

$(ONEAPI.tmpdir_y.dpc)/$(oneapi_y.dpc:%.$y=%_link.txt): \
    $(ONEAPI.objs_y.dpc) $(if $(OS_is_win),$(ONEAPI.tmpdir_y.dpc)/dll.res,) | $(ONEAPI.tmpdir_y.dpc)/. ; $(WRITE.PREREQS)
$(WORKDIR.lib)/$(oneapi_y.dpc): \
    $(daaldep.ipp) $(daaldep.vml) $(daaldep.mkl) \
    $(ONEAPI.tmpdir_y.dpc)/$(oneapi_y.dpc:%.$y=%_link.txt) ; $(DPC.LINK.DYNAMIC) ; $(LINK.DYNAMIC.POST)
$(WORKDIR.lib)/$(oneapi_y.dpc): LOPT += $(-fPIC)
$(WORKDIR.lib)/$(oneapi_y.dpc): LOPT += $(daaldep.rt.dpc)
$(WORKDIR.lib)/$(oneapi_y.dpc): LOPT += $(if $(OS_is_win),-IMPLIB:$(@:%.$(MAJORBINARY).dll=%_dll.lib),)
$(WORKDIR.lib)/$(oneapi_y.dpc): LOPT += $(if $(OS_is_win),$(WORKDIR.lib)/$(core_y:%.$(MAJORBINARY).dll=%_dll.lib))
$(WORKDIR.lib)/$(oneapi_y.dpc): LOPT += $(if $(OS_is_win),sycl.lib OpenCL.lib)
$(WORKDIR.lib)/$(oneapi_y.dpc): LOPT += $(mklgpufpk.LIBS_A)
ifdef OS_is_win
$(WORKDIR.lib)/$(oneapi_y.dpc:%.$(MAJORBINARY).dll=%_dll.lib): $(WORKDIR.lib)/$(oneapi_y.dpc)
endif

$(ONEAPI.tmpdir_y)/dll.res: $(VERSION_DATA_FILE)
$(ONEAPI.tmpdir_y)/dll.res: RCOPT += $(addprefix -I, $(WORKDIR) $(CORE.SERV.srcdir))
$(ONEAPI.tmpdir_y)/dll.res: $(CPPDIR.onedal)/dll.rc | $(ONEAPI.tmpdir_y)/. ; $(RC.COMPILE)

$(ONEAPI.tmpdir_y.dpc)/dll.res: $(VERSION_DATA_FILE)
$(ONEAPI.tmpdir_y.dpc)/dll.res: RCOPT += $(addprefix -I, $(WORKDIR) $(CORE.SERV.srcdir)) \
                                         -DONEDAL_DLL_RC_DATA_PARALLEL
$(ONEAPI.tmpdir_y.dpc)/dll.res: $(CPPDIR.onedal)/dll.rc | $(ONEAPI.tmpdir_y)/. ; $(RC.COMPILE)

#===============================================================================
# Threading parts
#===============================================================================
THR.srcs     := threading.cpp service_thread_pinner.cpp
THR.tmpdir_a := $(WORKDIR)/threading_static
THR.tmpdir_y := $(WORKDIR)/threading_dynamic
THR_TBB.objs_a := $(addprefix $(THR.tmpdir_a)/,$(THR.srcs:%.cpp=%_tbb.$o))
THR_TBB.objs_y := $(addprefix $(THR.tmpdir_y)/,$(THR.srcs:%.cpp=%_tbb.$o))
THR_SEQ.objs_a := $(addprefix $(THR.tmpdir_a)/,$(THR.srcs:%.cpp=%_seq.$o))
THR_SEQ.objs_y := $(addprefix $(THR.tmpdir_y)/,$(THR.srcs:%.cpp=%_seq.$o))
-include $(THR.tmpdir_a)/*.d
-include $(THR.tmpdir_y)/*.d

$(WORKDIR.lib)/$(thr_tbb_a): LOPT:=
$(WORKDIR.lib)/$(thr_tbb_a): $(THR_TBB.objs_a) $(daaldep.mkl.thr) ; $(LINK.STATIC)
$(WORKDIR.lib)/$(thr_seq_a): LOPT:=
$(WORKDIR.lib)/$(thr_seq_a): $(THR_SEQ.objs_a) $(daaldep.mkl.seq) ; $(LINK.STATIC)

$(THR.tmpdir_y)/%_link.def: $(THR.srcdir)/$(daaldep.$(PLAT).threxport) | $(THR.tmpdir_y)/.
	$(daaldep.$(_OS).threxport.create) > $@

$(WORKDIR.lib)/$(thr_tbb_y): LOPT += $(-fPIC) $(daaldep.rt.thr)
$(WORKDIR.lib)/$(thr_tbb_y): LOPT += $(if $(OS_is_win),-IMPLIB:$(@:%.dll=%_dll.lib),)
$(WORKDIR.lib)/$(thr_tbb_y): $(THR_TBB.objs_y) $(daaldep.mkl.thr) $(daaldep.mkl) $(if $(OS_is_win),$(THR.tmpdir_y)/dll_tbb.res,) $(THR.tmpdir_y)/$(thr_tbb_y:%.$y=%_link.def) ; $(LINK.DYNAMIC) ; $(LINK.DYNAMIC.POST)

$(WORKDIR.lib)/$(thr_seq_y): LOPT += $(-fPIC) $(daaldep.rt.seq)
$(WORKDIR.lib)/$(thr_seq_y): LOPT += $(if $(OS_is_win),-IMPLIB:$(@:%.dll=%_dll.lib),)
$(WORKDIR.lib)/$(thr_seq_y): $(THR_SEQ.objs_y) $(daaldep.mkl.seq) $(daaldep.mkl) $(if $(OS_is_win),$(THR.tmpdir_y)/dll_seq.res,) $(THR.tmpdir_y)/$(thr_seq_y:%.$y=%_link.def) ; $(LINK.DYNAMIC) ; $(LINK.DYNAMIC.POST)

THR.objs_a := $(THR_TBB.objs_a) $(THR_SEQ.objs_a)
THR.objs_y := $(THR_TBB.objs_y) $(THR_SEQ.objs_y)
THR_TBB.objs := $(THR_TBB.objs_a) $(THR_TBB.objs_y)
THR_SEQ.objs := $(THR_SEQ.objs_a) $(THR_SEQ.objs_y)
THR.objs := $(THR.objs_a) $(THR.objs_y)

$(THR.objs): COPT += $(-fPIC) $(-cxx11) $(-Zl) $(-DEBC) -DDAAL_HIDE_DEPRECATED -DTBB_USE_ASSERT=0
$(THR_TBB.objs): COPT += -D__DO_TBB_LAYER__
$(THR_SEQ.objs): COPT += -D__DO_SEQ_LAYER__

$(THR.objs_a): $(THR.tmpdir_a)/thr_inc_a_folders.txt
$(THR.objs_a): COPT += @$(THR.tmpdir_a)/thr_inc_a_folders.txt

$(THR.objs_y): $(THR.tmpdir_y)/thr_inc_y_folders.txt
$(THR.objs_y): COPT += @$(THR.tmpdir_y)/thr_inc_y_folders.txt
$(THR.objs_y): COPT += -D__DAAL_IMPLEMENTATION

$(THR.tmpdir_a)/thr_inc_a_folders.txt: makefile.lst | $(THR.tmpdir_a)/. $(CORE.incdirs) ; $(call WRITE.PREREQS,$(addprefix -I, $(CORE.incdirs)),$(space))
$(THR.tmpdir_y)/thr_inc_y_folders.txt: makefile.lst | $(THR.tmpdir_y)/. $(CORE.incdirs) ; $(call WRITE.PREREQS,$(addprefix -I, $(CORE.incdirs)),$(space))

$(THR_TBB.objs_a): $(THR.tmpdir_a)/%_tbb.$o: $(THR.srcdir)/%.cpp | $(THR.tmpdir_a)/. ; $(C.COMPILE)
$(THR_TBB.objs_y): $(THR.tmpdir_y)/%_tbb.$o: $(THR.srcdir)/%.cpp | $(THR.tmpdir_y)/. ; $(C.COMPILE)
$(THR_SEQ.objs_a): $(THR.tmpdir_a)/%_seq.$o: $(THR.srcdir)/%.cpp | $(THR.tmpdir_a)/. ; $(C.COMPILE)
$(THR_SEQ.objs_y): $(THR.tmpdir_y)/%_seq.$o: $(THR.srcdir)/%.cpp | $(THR.tmpdir_y)/. ; $(C.COMPILE)

$(THR.tmpdir_y)/dll_tbb.res: $(VERSION_DATA_FILE)
$(THR.tmpdir_y)/dll_seq.res: $(VERSION_DATA_FILE)
$(THR.tmpdir_y)/dll_tbb.res: RCOPT += -D_DAAL_THR_TBB $(addprefix -I, $(CORE.incdirs.common))
$(THR.tmpdir_y)/dll_seq.res: RCOPT += -D_DAAL_THR_SEQ $(addprefix -I, $(CORE.incdirs.common))

$(THR.tmpdir_y)/%_tbb.res: %.rc | $(THR.tmpdir_y)/. ; $(RC.COMPILE)
$(THR.tmpdir_y)/%_seq.res: %.rc | $(THR.tmpdir_y)/. ; $(RC.COMPILE)

#===============================================================================
# Java/JNI part
#===============================================================================
JAVA.srcdir      := $(DIR)/java
JAVA.srcdir.full := $(JAVA.srcdir)/com/intel/daal
JAVA.tmpdir      := $(WORKDIR)/java_tmpdir

JNI.srcdir       := $(DIR)/java
JNI.srcdir.full  := $(JNI.srcdir)/com/intel/daal
JNI.tmpdir       := $(WORKDIR)/jni_tmpdir

JAVA.srcdirs := $(JAVA.srcdir.full)                                                                                         \
                $(JAVA.srcdir.full)/algorithms $(addprefix $(JAVA.srcdir.full)/algorithms/,$(JJ.ALGORITHMS))                \
                $(JAVA.srcdir.full)/data_management $(addprefix $(JAVA.srcdir.full)/data_management/,$(JJ.DATA_MANAGEMENT)) \
                $(JAVA.srcdir.full)/services \
				$(JAVA.srcdir.full)/utils
JAVA.srcs.f := $(wildcard $(JAVA.srcdirs:%=%/*.java))
JAVA.srcs   := $(subst $(JAVA.srcdir)/,,$(JAVA.srcs.f))

JNI.srcdirs := $(JNI.srcdir.full)                                                                                        \
               $(JNI.srcdir.full)/algorithms $(addprefix $(JNI.srcdir.full)/algorithms/,$(JJ.ALGORITHMS))                \
               $(JNI.srcdir.full)/data_management $(addprefix $(JNI.srcdir.full)/data_management/,$(JJ.DATA_MANAGEMENT)) \
               $(JNI.srcdir.full)/services \
			   $(JNI.srcdir.full)/utils
JNI.srcs.f := $(wildcard $(JNI.srcdirs:%=%/*.cpp))
JNI.srcs   := $(subst $(JNI.srcdir)/,,$(JNI.srcs.f))
JNI.objs   := $(addprefix $(JNI.tmpdir)/,$(JNI.srcs:%.cpp=%.$o))

-include $(if $(wildcard $(JNI.tmpdir)/*),$(shell find $(JNI.tmpdir) -name "*.d"))

#----- production of $(daal_jar)
# javac does not generate dependences. Therefore we pass all *.java files to
# a single launch of javac and let it resolve dependences on its own.
# TODO: create hierarchy in java/jni temp folders madually
$(WORKDIR.lib)/$(daal_jar:%.jar=%_jar_link.txt): $(JAVA.srcs.f) | $(WORKDIR.lib)/. ; $(WRITE.PREREQS)
$(WORKDIR.lib)/$(daal_jar):                  $(WORKDIR.lib)/$(daal_jar:%.jar=%_jar_link.txt)
	rm -rf $(JAVA.tmpdir) ; mkdir -p $(JAVA.tmpdir)
	mkdir -p $(JNI.tmpdir)
	javac -classpath $(JAVA.tmpdir) $(-DEBJ) -d $(JAVA.tmpdir) -h $(JNI.tmpdir) @$(WORKDIR.lib)/$(daal_jar:%.jar=%_jar_link.txt)
	jar cvf $@ -C $(JAVA.tmpdir) .

#----- production of JNI dll
$(WORKDIR.lib)/$(jni_so): LOPT += $(-fPIC)
$(WORKDIR.lib)/$(jni_so): LOPT += $(daaldep.rt.thr) $(daaldep.mkl.thr)
$(JNI.tmpdir)/$(jni_so:%.$y=%_link.txt): $(JNI.objs) $(if $(OS_is_win),$(JNI.tmpdir)/dll.res,) $(WORKDIR.lib)/$(core_a) $(WORKDIR.lib)/$(thr_tbb_a) ; $(WRITE.PREREQS)
$(WORKDIR.lib)/$(jni_so):                $(JNI.tmpdir)/$(jni_so:%.$y=%_link.txt); $(LINK.DYNAMIC) ; $(LINK.DYNAMIC.POST)

$(JNI.objs): $(JNI.tmpdir)/inc_j_folders.txt
$(JNI.objs): $(WORKDIR.lib)/$(daal_jar)
$(JNI.objs): COPT += $(-fPIC) $(-cxx11) $(-Zl) $(-DEBC) -DDAAL_NOTHROW_EXCEPTIONS -DDAAL_HIDE_DEPRECATED -DTBB_USE_ASSERT=0
$(JNI.objs): COPT += @$(JNI.tmpdir)/inc_j_folders.txt

$(JNI.tmpdir)/inc_j_folders.txt: makefile.lst | $(JNI.tmpdir)/. ; $(call WRITE.PREREQS,$(addprefix -I,$(JNI.tmpdir) $(CORE.incdirs.common) $(CORE.incdirs.thirdp) $(JNI.srcdir)),$(space))

$(JNI.objs): $(JNI.tmpdir)/%.$o: $(JNI.srcdir)/%.cpp; mkdir -p $(@D); $(C.COMPILE)

$(JNI.tmpdir)/dll.res: $(VERSION_DATA_FILE)
$(JNI.tmpdir)/dll.res: RCOPT += -D_DAAL_JAVA_INTERF $(addprefix -I, $(CORE.incdirs.common))
$(JNI.tmpdir)/%.res: %.rc | $(JNI.tmpdir)/. ; $(RC.COMPILE)

#===============================================================================
# Top level targets
#===============================================================================
daal: $(if $(CORE.ALGORITHMS.CUSTOM),           \
          _daal _release_c,                     \
          _daal _daal_jj _release _release_doc  \
      )
daal_c: _daal _release_c

oneapi: oneapi_c oneapi_dpc
oneapi_c: _oneapi_c _release_oneapi_c
oneapi_dpc: _oneapi_dpc _release_oneapi_dpc

onedal: oneapi daal
onedal_c: daal_c oneapi_c
onedal_dpc: daal_c oneapi_c oneapi_dpc

_daal:    _daal_core _daal_thr
_daal_jj: _daal_jar _daal_jni

_daal_core:  info.building.core
_daal_core:  $(WORKDIR.lib)/$(core_a) $(WORKDIR.lib)/$(core_y) ## TODO: move list of needed libs to one env var!!!
_daal_thr:   info.building.threading
_daal_thr:   $(if $(DAALTHRS),$(foreach ithr,$(DAALTHRS),_daal_thr_$(ithr)),)
_daal_thr_tbb:   $(WORKDIR.lib)/$(thr_tbb_a) $(WORKDIR.lib)/$(thr_tbb_y)
_daal_thr_seq:   $(WORKDIR.lib)/$(thr_seq_a) $(WORKDIR.lib)/$(thr_seq_y)
_daal_jar _daal_jni: info.building.java
_daal_jar: $(WORKDIR.lib)/$(daal_jar)
_daal_jni: $(WORKDIR.lib)/$(jni_so)

_release:    _release_c _release_jj
_release_c:  _release_c_h _release_common
_release_jj: _release_common

_oneapi_c: info.building.oneapi.C++.part
_oneapi_c: $(WORKDIR.lib)/$(oneapi_a) $(WORKDIR.lib)/$(oneapi_y)

_oneapi_dpc: info.building.oneapi.DPC++.part
_oneapi_dpc: $(WORKDIR.lib)/$(oneapi_a.dpc)

_release_oneapi_c: _release_oneapi_c_h _release_oneapi_common
_release_oneapi_dpc: _release_oneapi_c _release_oneapi_common

#-------------------------------------------------------------------------------
# Populating RELEASEDIR
#-------------------------------------------------------------------------------
upd = $(cpy)

_release: info.building.release

#----- releasing static and dynamic libraries
define .release.y_win
$3: $2/$1
$(if $(phony-upd),$(eval .PHONY: $2/$1))
$2/$1: $(WORKDIR.lib)/$1 | $2/.
	cp -fp $(WORKDIR.lib)/$1 $2/$1
endef
define .release.a_win
$3: $2/$1
$(if $(phony-upd),$(eval .PHONY: $2/$1))
$2/$1: $(WORKDIR.lib)/$1 | $2/.
	cp -fp $(WORKDIR.lib)/$1 $2/$1
ifneq (,$(findstring dll.,$1))
	cp -fp $(WORKDIR.lib)/$1 $2/$(subst dll.,dll.$(MAJORBINARY).,$1)
endif
endef
define .release.y_link
$3: $2/$1
$(if $(phony-upd),$(eval .PHONY: $2/$1))
$2/$1: $(WORKDIR.lib)/$1 | $2/.
	cp -fp $(WORKDIR.lib)/$1 $2/$(subst .$y,$(y_full_name_postfix),$1) && cd $2 && ln -sf $(subst .$y,$(y_full_name_postfix),$1) $(subst .$y,$(y_major_name_postfix),$1) && ln -sf $(subst .$y,$(y_major_name_postfix),$1) $1
endef
define .release.a
$3: $2/$1
$(if $(phony-upd),$(eval .PHONY: $2/$1))
$2/$1: $(WORKDIR.lib)/$1 | $2/. ; $(value upd)
endef

ifeq ($(if $(or $(OS_is_lnx),$(OS_is_mac)),yes,),yes)
$(foreach x,$(release.LIBS_A),$(eval $(call .release.a,$x,$(RELEASEDIR.libia),_release_c)))
$(foreach x,$(release.LIBS_Y),$(eval $(call .release.y_link,$x,$(RELEASEDIR.soia),_release_c)))
$(foreach x,$(release.LIBS_J),$(eval $(call .release.y_link,$x,$(RELEASEDIR.soia),_release_jj)))
$(foreach x,$(release.ONEAPI.LIBS_A),$(eval $(call .release.a,$x,$(RELEASEDIR.libia),_release_oneapi_c)))
$(foreach x,$(release.ONEAPI.LIBS_Y),$(eval $(call .release.y_link,$x,$(RELEASEDIR.soia),_release_oneapi_c)))
$(foreach x,$(release.ONEAPI.LIBS_A.dpc),$(eval $(call .release.a,$x,$(RELEASEDIR.libia),_release_oneapi_dpc)))
$(foreach x,$(release.ONEAPI.LIBS_Y.dpc),$(eval $(call .release.y_link,$x,$(RELEASEDIR.soia),_release_oneapi_dpc)))
endif

ifeq ($(OS_is_win),yes)
$(foreach x,$(release.LIBS_A),$(eval $(call .release.a_win,$x,$(RELEASEDIR.libia),_release_c)))
$(foreach x,$(release.LIBS_Y),$(eval $(call .release.y_win,$x,$(RELEASEDIR.soia),_release_c)))
$(foreach x,$(release.LIBS_J),$(eval $(call .release.a_win,$x,$(RELEASEDIR.soia),_release_jj)))
$(foreach x,$(release.ONEAPI.LIBS_A),$(eval $(call .release.a_win,$x,$(RELEASEDIR.libia),_release_oneapi_c)))
$(foreach x,$(release.ONEAPI.LIBS_Y),$(eval $(call .release.y_win,$x,$(RELEASEDIR.soia),_release_oneapi_c)))
$(foreach x,$(release.ONEAPI.LIBS_A.dpc),$(eval $(call .release.a_win,$x,$(RELEASEDIR.libia),_release_oneapi_dpc)))
$(foreach x,$(release.ONEAPI.LIBS_Y.dpc),$(eval $(call .release.y_win,$x,$(RELEASEDIR.soia),_release_oneapi_dpc)))
endif

ifneq ($(MKLGPUFPKDIR),)
# Copies the file to the destination directory and renames daal -> onedal
# $1: Path to the file to be copied
# $2: Destination directory
define .release.sycl.old
_release_common: $2/$(subst daal_sycl.$a,onedal_sycl.$a,$(notdir $1))
$2/$(subst daal_sycl.$a,onedal_sycl.$a,$(notdir $1)): $(call frompf1,$1) | $2/. ; $(value cpy)
endef

$(foreach t,$(mklgpufpk.HEADERS),$(eval $(call .release.sycl.old,$t,$(RELEASEDIR.include.mklgpufpk))))
$(foreach t,$(mklgpufpk.LIBS_A), $(eval $(call .release.sycl.old,$t,$(RELEASEDIR.libia))))
endif

#----- releasing jar files
_release_jj: $(addprefix $(RELEASEDIR.jardir)/,$(release.JARS))
$(RELEASEDIR.jardir)/%.jar: $(WORKDIR.lib)/%.jar | $(RELEASEDIR.jardir)/. ; $(cpy)

#----- releasing examples
define .release.x
$3: $2/$(subst _$(_OS),,$1)
$2/$(subst _$(_OS),,$1): $(DIR)/$1 | $(dir $2/$1)/.
	$(if $(filter %makefile_win,$1),python ./deploy/local/generate_win_makefile.py $(dir $(DIR)/$1) $(dir $2/$1),$(value cpy))
	$(if $(filter %.sh %.bat,$1),chmod +x $$@)
endef
$(foreach x,$(release.EXAMPLES.DATA),$(eval $(call .release.x,$x,$(RELEASEDIR.daal),_release_common)))
$(foreach x,$(release.EXAMPLES.CPP),$(eval $(call .release.x,$x,$(RELEASEDIR.daal),_release_c)))
$(foreach x,$(release.EXAMPLES.JAVA),$(eval $(call .release.x,$x,$(RELEASEDIR.daal),_release_jj)))
$(foreach x,$(release.ONEAPI.EXAMPLES.CPP),$(eval $(call .release.x,$x,$(RELEASEDIR.daal),_release_oneapi_c)))
$(foreach x,$(release.ONEAPI.EXAMPLES.DPC),$(eval $(call .release.x,$x,$(RELEASEDIR.daal),_release_oneapi_dpc)))
$(foreach x,$(release.ONEAPI.EXAMPLES.DATA),$(eval $(call .release.x,$x,$(RELEASEDIR.daal),_release_oneapi_common)))
$(foreach x,$(release.EXAMPLES.COMMON_CMAKE),$(eval $(call .release.x,$x,$(RELEASEDIR.daal),_release_common)))

#----- releasing VS solutions
ifeq ($(OS_is_win),yes)
# $1: Relative examples directiry
# $2: Solution filename
# $3: Prefix of solution template
# $4: Target to trigger
define .release.x.sln
$4: $(RELEASEDIR.daal)/$1/$2
$(RELEASEDIR.daal)/$1/$2: \
        $1/$3.vcxproj.tpl \
        $1/$3.vcxproj.filters.tpl \
        $1/$3.vcxproj.user.tpl \
        $1/$3.sln.tpl
	python ./deploy/local/generate_win_solution.py $1 $(RELEASEDIR.daal)/$1 $2 --template_name $3
endef

$(eval $(call .release.x.sln,examples/daal/cpp,DAALExamples.sln,daal_win,_release_c))
$(eval $(call .release.x.sln,examples/daal/cpp_sycl,DAALExamples_sycl.sln,daal_win,_release_c))
$(eval $(call .release.x.sln,examples/oneapi/cpp,oneDALExamples.sln,onedal_win,_release_oneapi_c))
$(eval $(call .release.x.sln,examples/oneapi/dpc,oneDALExamples.sln,onedal_win,_release_oneapi_dpc))
endif

#----- releasing environment scripts
define .release.x
$4: $3/$2
$3/$2: $(DIR)/$1 | $3/. ; $(value cpy)
	$(if $(filter %.sh %.bat dal,$2),sed -i -e 's/__DAL_MAJOR_BINARY__/$(MAJORBINARY)/' $3/$2)
	$(if $(filter %.sh %.bat dal,$2),sed -i -e 's/__DAL_MINOR_BINARY__/$(MINORBINARY)/' $3/$2)
	$(if $(OS_is_win),unix2dos $3/$2)
	$(if $(filter %.sh %.bat,$2),chmod +x $$@)
endef
$(foreach x,$(release.ENV),$(eval $(call .release.x,$x,$(notdir $(subst _$(_OS),,$x)),$(RELEASEDIR.env),_release_common)))
$(if $(OS_is_lnx),$(foreach x,$(release.MODULEFILES),$(eval $(call .release.x,$x,$(notdir $x),$(RELEASEDIR.modulefiles),_release_common))))
$(foreach x,$(release.CONF),$(eval $(call .release.x,$x,$(notdir $(subst _$(_OS),,$x)),$(RELEASEDIR.conf),_release_common)))

#----- releasing documentation
_release_doc:
define .release.d
_release_doc: $2
$2: $1 | $(dir $2)/. ; $(value cpy)
	$(if $(filter %.sh %.bat,$2),chmod +x $$@)
endef
$(foreach d,$(release.DOC.COMMON),    $(eval $(call .release.d,$d,$(subst $(DOC.srcdir),    $(RELEASEDIR.doc),    $(subst _$(_OS),,$d)))))
$(foreach d,$(release.DOC.OSSPEC),    $(eval $(call .release.d,$d,$(subst $(DOC.srcdir),    $(RELEASEDIR.doc),    $(subst _$(_OS),,$d)))))

#----- releasing samples and headers
define .release.d
$3: $2
$2: $1 | $(dir $2)/. ; $(value cpy)
	$(if $(filter %.sh %.bat,$2),chmod +x $$@)
endef
$(foreach d,$(release.SAMPLES.CPP),   $(eval $(call .release.d,$d,$(subst $(SAMPLES.srcdir),$(RELEASEDIR.samples),$(subst _$(_OS),,$d)),_release_c)))
$(foreach d,$(release.SAMPLES.JAVA),  $(eval $(call .release.d,$d,$(subst $(SAMPLES.srcdir),$(RELEASEDIR.samples),$(subst _$(_OS),,$d)),_release_jj)))
$(foreach d,$(release.SAMPLES.SCALA), $(eval $(call .release.d,$d,$(subst $(SAMPLES.srcdir),$(RELEASEDIR.samples),$(subst _$(_OS),,$d)),_release_jj)))

$(CORE.incdirs): _release_c_h

define .release.dd
$3: $2
$2: $1 ; $(value mkdir)$(value cpy)
	$(if $(filter %library_version_info.h,$2),+$(daalmake) -f makefile update_headers_version)
	$(if $(USECPUS.out.defs.filter),$(if $(filter %daal_kernel_defines.h,$2),$(USECPUS.out.defs.filter) $2; rm -rf $(subst .h,.h.bak,$2)))
endef
$(foreach d,$(release.HEADERS.COMMON),$(eval $(call .release.dd,$d,$(subst $(CPPDIR.daal)/include/,$(RELEASEDIR.include)/,$d),_release_c_h)))
$(foreach d,$(release.HEADERS.OSSPEC),$(eval $(call .release.dd,$d,$(subst $(CPPDIR.daal)/include/,$(RELEASEDIR.include)/,$(subst _$(_OS),,$d)),_release_c_h)))

define .release.oneapi.dd
$3: $2
$2: $1 ; $(value mkdir)$(value cpy)
endef
$(foreach d,$(release.ONEAPI.HEADERS.COMMON),$(eval $(call .release.oneapi.dd,$d,$(subst $(CPPDIR)/,$(RELEASEDIR.include)/,$d),_release_oneapi_c_h)))
$(foreach d,$(release.ONEAPI.HEADERS.OSSPEC),$(eval $(call .release.oneapi.dd,$d,$(subst $(CPPDIR)/,$(RELEASEDIR.include)/,$(subst _$(_OS),,$d)),_release_oneapi_c_h)))

#----- releasing static/dynamic Intel(R) TBB libraries
$(RELEASEDIR.tbb.libia) $(RELEASEDIR.tbb.soia): _release_common

define .release.t
_release_common: $2/$(notdir $1)
$2/$(notdir $1): $(call frompf1,$1) | $2/. ; $(value cpy)
endef
$(foreach t,$(releasetbb.LIBS_Y),$(eval $(call .release.t,$t,$(RELEASEDIR.tbb.soia))))
$(foreach t,$(releasetbb.LIBS_A),$(eval $(call .release.t,$t,$(RELEASEDIR.tbb.libia))))

#===============================================================================
# Miscellaneous stuff
#===============================================================================

.PHONY: clean cleanrel cleanall
clean:    ; -rm -rf $(WORKDIR)
cleanrel: ; -rm -rf $(RELEASEDIR)
cleanall: clean cleanrel

define help
Usage: make [target...] [flag=value...]
Targets:
  daal      - build all (use -j to speedup the build)
  _daal_core ... _daal_jar _daal_jni - build only a part of the product,
             without populating release directory (read makefile for details)
  _release - populate release directory
  clean    - clean working directory $(WORKDIR)
  cleanrel - clean release directory $(RELEASEDIR) (for entire OS!)
  cleanall - clean both working and release directories
Flags:
  COMPILER   - compiler to use ($(COMPILERs)) [default: $(COMPILER)]
  WORKDIR    - directory for intermediate results [default: $(WORKDIR)]
  RELEASEDIR - directory for release [default: $(RELEASEDIR)]
  CORE.ALGORITHMS.CUSTOM - list of algorithms to be included into library
      build cpp interfaces only
      possible values: $(CORE.ALGORITHMS.CUSTOM.AVAILABLE)
  REQCPU - list of CPU optimizations to be included into library
      possible values: $(CPUs)
  REQDBG - Flag that enables build in debug mode
endef

daal_dbg:
	@echo "1" "!$(mklgpufpk.LIBS_A)!"
	@echo "2" "!$(MKLGPUFPKDIR)!"
	@echo "3" "!$(MKLGPUFPKROOT)!"
