#===============================================================================
# Copyright 2014-2019 Intel Corporation
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

COMPILERs = icc gnu clang vc
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
attr.lnx32  = lnx ia32    lin
attr.mac32e = mac intel64
attr.win32e = win intel64 win
attr.win32  = win ia32    win
attr.fbsd32e = fbsd intel64 fre
attr.fbsd32  = fbsd ia32    fre

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

include build/cmplr.$(COMPILER).mk

$(if $(filter $(PLATs.$(COMPILER)),$(PLAT)),,$(error PLAT for $(COMPILER) must be defined to one of $(PLATs.$(COMPILER))))

#===============================================================================
# Dependencies generation
#===============================================================================

include build/common.mk
include build/deps.mk

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
-fPIC  := $(if $(OS_is_win),,-fPIC)
-Zl    := $(-Zl.$(COMPILER))
-DEBC  := $(if $(REQDBG),$(-DEBC.$(COMPILER)) -DDEBUG_ASSERT) -DTBB_SUPPRESS_DEPRECATED_MESSAGES -D__TBB_LEGACY_MODE $(if $(REQPRF), -D__DAAL_ITTNOTIFY_ENABLE__)
-DEBJ  := $(if $(REQDBG),-g,-g:none)
-DEBL  := $(if $(REQDBG),$(if $(OS_is_win),-debug,))
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
# daal/lib/ia32 - static and dynamic libraries for ia32
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
# daal/lib/ia32 - static and import libraries for ia32
# daal/lib/intel64 - static and import libraries for intel64
# redist/ia32/daal - dlls for ia32
# redist/intel64/daal - dlls for intel64

# FREEBSD release structure (under __release_fbsd):
# daal
# daal/bin - platform independent binaries: env setters
# daal/examples - usage demonstrations
# daal/include - header files
# daal/lib - platform-independent libraries (jar files)
# daal/lib/ia32 - static and dynamic libraries for ia32
# daal/lib/intel64 - static and dynamic libraries for intel64

# List of needed threadings layers can be specified in DAALTHRS.
# if DAALTHRS is empty, threading will be incapsulated to core
DAALTHRS ?= tbb seq
DAALAY   ?= a y

DIR:=.
WORKDIR    ?= $(DIR)/__work$(CMPLRDIRSUFF.$(COMPILER))/$(PLAT)
RELEASEDIR ?= $(DIR)/__release_$(_OS)$(CMPLRDIRSUFF.$(COMPILER))
RELEASEDIR.daal    := $(RELEASEDIR)/daal/latest
RELEASEDIR.lib     := $(RELEASEDIR.daal)/lib
RELEASEDIR.env     := $(RELEASEDIR.daal)/env
RELEASEDIR.conf    := $(RELEASEDIR.daal)/config
RELEASEDIR.doc     := $(RELEASEDIR.daal)/documentation
RELEASEDIR.samples := $(RELEASEDIR.daal)/samples
RELEASEDIR.jardir  := $(RELEASEDIR.daal)/lib
RELEASEDIR.libia   := $(RELEASEDIR.daal)/lib$(if $(OS_is_mac),,/$(_IA))
RELEASEDIR.include := $(RELEASEDIR.daal)/include
RELEASEDIR.soia    := $(if $(OS_is_win),$(RELEASEDIR.daal)/redist/$(_IA),$(RELEASEDIR.libia))
WORKDIR.lib := $(WORKDIR)/daal/lib

COVFILE   := $(subst BullseyeStub,$(RELEASEDIR.daal)/Bullseye_$(_IA).cov,$(COVFILE))
COV.libia := $(if $(BULLSEYEROOT),$(BULLSEYEROOT)/lib)

MKLFPKDIR:= $(if $(wildcard $(DIR)/externals/mklfpk/$(_OS)/*),$(DIR)/externals/mklfpk,                            \
                $(if $(wildcard $(MKLFPKROOT)/include/*),$(subst \,/,$(MKLFPKROOT)),                              \
                    $(error Can`t find MKLFPK libs nether in $(DIR)/externals/mklfpk/$(_OS) not in MKLFPKROOT.)))
MKLFPKDIR.include := $(MKLFPKDIR)/include $(MKLFPKDIR)/$(if $(OS_is_fbsd),lnx,$(_OS))/include
MKLFPKDIR.libia   := $(MKLFPKDIR)/$(if $(OS_is_fbsd),lnx,$(_OS))/lib/$(_IA)

topf = $(shell echo $1 | sed 's/Program Files (x86)/ProgramFilesx86/' | sed 's/\\/\//g' )
frompf = $(shell echo $1 | sed 's/ProgramFilesx86/Program\\ Files\\ (x86)/')


#============================= TBB folders =====================================
TBBDIR := $(if $(wildcard $(DIR)/externals/tbb/$(_OS)/*),$(DIR)/externals/tbb/$(_OS)$(if $(OS_is_win),/tbb))
TBBDIR.2 := $(if $(TBBDIR),$(TBBDIR),$(call topf,$$TBBROOT))
TBBDIR.2 := $(if $(TBBDIR.2),$(TBBDIR.2),$(error Can`t find TBB neither in $(DIR)/externals/tbb nor in $$TBBROOT))

TBBDIR.include := $(if $(TBBDIR),$(TBBDIR)/include/tbb $(TBBDIR)/include)

TBBDIR.libia.prefix := $(TBBDIR.2)/lib

TBBDIR.libia.win.vc1  := $(if $(OS_is_win),$(if $(wildcard $(TBBDIR.libia.prefix)/$(_IA)/vc_mt),$(TBBDIR.libia.prefix)/$(_IA)/vc_mt))
TBBDIR.libia.win.vc2  := $(if $(OS_is_win),$(if $(TBBDIR.libia.win.vc1),,$(firstword $(filter $(TBBROOT)%,$(subst ;,$(space),$(LIB))))))
TBBDIR.libia.win.vc22 := $(if $(OS_is_win),$(if $(TBBDIR.libia.win.vc2),$(wildcard $(TBBDIR.libia.win.vc2)/tbb.dll)))
TBBDIR.libia.win:= $(if $(OS_is_win),$(if $(TBBDIR.libia.win.vc22),$(TBBDIR.libia.win.vc2),$(if $(TBBDIR.libia.win.vc1),$(TBBDIR.libia.win.vc1),$(error Can`t find TBB libs nether in $(TBBDIR.libia.prefix)/$(_IA)/vc_mt not in $(firstword $(filter $(TBBROOT)%,$(subst ;,$(space),$(LIB)))).))))
TBBDIR.libia.win:= $(subst \,/,$(TBBDIR.libia.win))

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
TBBDIR.soia.win  := $(if $(OS_is_win),$(if $(TBBDIR.libia.win.vc22),$(TBBDIR.libia.win.vc2),$(if $(wildcard $(TBBDIR.soia.prefix)/redist/$(_IA)/vc_mt/*),$(TBBDIR.soia.prefix)/redist/$(_IA)/vc_mt,$(error Can`t find TBB runtimes nether in $(TBBDIR.soia.prefix)/redist/$(_IA)/vc_mt not in $(firstword $(filter $(TBBROOT)%,$(subst ;,$(space),$(LIB)))).))))
TBBDIR.soia.win  := $(subst \,/,$(TBBDIR.soia.win))
TBBDIR.soia.lnx  := $(if $(OS_is_lnx),$(TBBDIR.libia.lnx))
TBBDIR.soia.mac  := $(if $(OS_is_mac),$(TBBDIR.libia.mac))
TBBDIR.soia.fbsd := $(if $(OS_is_fbsd),$(TBBDIR.soia.prefix)/lib)
TBBDIR.soia := $(TBBDIR.soia.$(_OS))

RELEASEDIR.tbb       := $(RELEASEDIR)/tbb/latest
RELEASEDIR.tbb.libia := $(RELEASEDIR.tbb)/lib$(if $(OS_is_mac),,/$(_IA)$(if $(OS_is_win),/vc_mt,/$(TBBDIR.libia.lnx.gcc)))
RELEASEDIR.tbb.soia  := $(if $(OS_is_win),$(RELEASEDIR.tbb)/redist/$(_IA)/vc_mt,$(RELEASEDIR.tbb.libia))
releasetbb.LIBS_A := $(if $(OS_is_win),$(TBBDIR.libia)/tbb.$(a) $(TBBDIR.libia)/tbbmalloc.$(a))
releasetbb.LIBS_Y := $(TBBDIR.soia)/$(plib)tbb.$(y) $(TBBDIR.soia)/$(plib)tbbmalloc.$(y) \
                     $(if $(or $(OS_is_lnx),$(OS_is_fbsd)),$(TBBDIR.libia)/$(plib)tbb.so.2 $(TBBDIR.libia)/$(plib)tbbmalloc.so.2,)

RELEASEDIR.include.mklgpufpk := $(RELEASEDIR.include)/oneapi/internal/math

MKLGPUFPKDIR:= $(if $(wildcard $(DIR)/externals/mklgpufpk/*),$(DIR)/externals/mklgpufpk,$(subst \,/,$(MKLGPUFPKROOT)))
MKLGPUFPKDIR.include := $(MKLGPUFPKDIR)/include
MKLGPUFPKDIR.libia   := $(MKLGPUFPKDIR)/lib/$(_IA)

mklgpufpk.LIBS_A := $(MKLGPUFPKDIR.libia)/$(plib)daal_sycl.$(a)
mklgpufpk.HEADERS := $(MKLGPUFPKDIR.include)/mkl_dal_sycl.hpp $(MKLGPUFPKDIR.include)/mkl_dal_blas_sycl.hpp

#===============================================================================
# Release library names
#===============================================================================

core_a    := $(plib)daal_core.$a
core_y    := $(plib)daal_core.$y

thr_tbb_a := $(plib)daal_thread.$a
thr_seq_a := $(plib)daal_sequential.$a
thr_tbb_y := $(plib)daal_thread.$y
thr_seq_y := $(plib)daal_sequential.$y

daal_jar  := daal.jar

jni_so    := $(plib)JavaAPI.$y

release.LIBS_A := $(core_a)                                                             \
                  $(if $(OS_is_win),$(foreach ilib,$(core_a),$(ilib:%.lib=%_dll.lib)),) \
                  $(if $(DAALTHRS),$(foreach i,$(DAALTHRS),$(thr_$(i)_a)),)
release.LIBS_Y := $(core_y) $(if $(DAALTHRS),$(foreach i,$(DAALTHRS),$(thr_$(i)_y)),)
release.LIBS_J := $(jni_so)
release.JARS = $(daal_jar)

# Libraries required for building
daaldep.lnx32e.mkl.thr := $(MKLFPKDIR.libia)/$(plib)daal_mkl_thread.$a
daaldep.lnx32e.mkl.seq := $(MKLFPKDIR.libia)/$(plib)daal_mkl_sequential.$a
daaldep.lnx32e.mkl := $(MKLFPKDIR.libia)/$(plib)daal_vmlipp_core.$a
daaldep.lnx32e.vml := 
daaldep.lnx32e.ipp := $(if $(COV.libia),$(COV.libia)/libcov.a)
daaldep.lnx32e.rt.thr := -L$(RELEASEDIR.tbb.soia) -ltbb -ltbbmalloc -lpthread $(daaldep.lnx32e.rt.$(COMPILER)) $(if $(COV.libia),$(COV.libia)/libcov.a)
daaldep.lnx32e.rt.seq := -lpthread $(daaldep.lnx32e.rt.$(COMPILER)) $(if $(COV.libia),$(COV.libia)/libcov.a)
daaldep.lnx32e.threxport := export_lnx32e.def

daaldep.lnx32.mkl.thr := $(MKLFPKDIR.libia)/$(plib)daal_mkl_thread.$a    
daaldep.lnx32.mkl.seq := $(MKLFPKDIR.libia)/$(plib)daal_mkl_sequential.$a
daaldep.lnx32.mkl := $(MKLFPKDIR.libia)/$(plib)daal_vmlipp_core.$a
daaldep.lnx32.vml := 
daaldep.lnx32.ipp := $(if $(COV.libia),$(COV.libia)/libcov32.a)
daaldep.lnx32.rt.thr := -L$(RELEASEDIR.tbb.soia) -ltbb -ltbbmalloc -lpthread $(daaldep.lnx32.rt.$(COMPILER)) $(if $(COV.libia),$(COV.libia)/libcov32.a)
daaldep.lnx32.rt.seq := -lpthread $(daaldep.lnx32.rt.$(COMPILER)) $(if $(COV.libia),$(COV.libia)/libcov32.a)
daaldep.lnx32.threxport := export.def

daaldep.lnx.threxport.create = grep -v -E '^(EXPORTS|;|$$)' $< $(USECPUS.out.grep.filter) | sed -e 's/^/-u /'


daaldep.win32e.mkl.thr := $(MKLFPKDIR.libia)/daal_mkl_thread.$a
daaldep.win32e.mkl.seq := $(MKLFPKDIR.libia)/daal_mkl_sequential.$a
daaldep.win32e.mkl := $(MKLFPKDIR.libia)/$(plib)daal_vmlipp_core.$a
daaldep.win32e.vml := 
daaldep.win32e.ipp := 
daaldep.win32e.rt.thr  := -LIBPATH:$(RELEASEDIR.tbb.libia) tbb.lib tbbmalloc.lib libcpmt.lib libcmt.lib $(if $(CHECK_DLL_SIG),Wintrust.lib)
daaldep.win32e.rt.seq  := libcpmt.lib libcmt.lib $(if $(CHECK_DLL_SIG),Wintrust.lib)
daaldep.win32e.threxport := export_lnx32e.def

daaldep.win32.mkl.thr := $(MKLFPKDIR.libia)/daal_mkl_thread.$a
daaldep.win32.mkl.seq := $(MKLFPKDIR.libia)/daal_mkl_sequential.$a
daaldep.win32.mkl := $(MKLFPKDIR.libia)/$(plib)daal_vmlipp_core.$a
daaldep.win32.vml := 
daaldep.win32.ipp := 
daaldep.win32.rt.thr := -LIBPATH:$(RELEASEDIR.tbb.libia) tbb.lib tbbmalloc.lib libcpmt.lib libcmt.lib $(if $(CHECK_DLL_SIG),Wintrust.lib)
daaldep.win32.rt.seq := libcpmt.lib libcmt.lib $(if $(CHECK_DLL_SIG),Wintrust.lib)
daaldep.win32.threxport := export.def

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

daaldep.fbsd32.mkl.thr := $(MKLFPKDIR.libia)/$(plib)daal_mkl_thread.$a
daaldep.fbsd32.mkl.seq := $(MKLFPKDIR.libia)/$(plib)daal_mkl_sequential.$a
daaldep.fbsd32.mkl := $(MKLFPKDIR.libia)/$(plib)daal_vmlipp_core.$a
daaldep.fbsd32.vml :=
daaldep.fbsd32.ipp := $(if $(COV.libia),$(COV.libia)/libcov32.a)
daaldep.fbsd32.rt.thr := -L$(RELEASEDIR.tbb.soia) -ltbb -ltbbmalloc -lpthread $(daaldep.fbsd32.rt.$(COMPILER)) $(if $(COV.libia),$(COV.libia)/libcov32.a)
daaldep.fbsd32.rt.seq := -lpthread $(daaldep.fbsd32.rt.$(COMPILER)) $(if $(COV.libia),$(COV.libia)/libcov32.a)
daaldep.fbsd32.threxport := export.def

daaldep.fbsd.threxport.create = grep -v -E '^(EXPORTS|;|$$)' $< $(USECPUS.out.grep.filter) | sed -e 's/^/-Wl,-u -Wl,/'


daaldep.mkl.thr := $(daaldep.$(PLAT).mkl.thr)
daaldep.mkl.seq := $(daaldep.$(PLAT).mkl.seq)
daaldep.mkl     := $(daaldep.$(PLAT).mkl)
daaldep.vml     := $(daaldep.$(PLAT).vml)
daaldep.ipp     := $(daaldep.$(PLAT).ipp)
daaldep.rt.thr  := $(daaldep.$(PLAT).rt.thr)
daaldep.rt.seq  := $(daaldep.$(PLAT).rt.seq)

# List header files to populate release/include.
release.HEADERS := $(shell find include -type f -name "*.h")
release.HEADERS.OSSPEC := $(foreach fn,$(release.HEADERS),$(if $(filter %$(_OS),$(basename $(fn))),$(fn)))
release.HEADERS.COMMON := $(foreach fn,$(release.HEADERS),$(if $(filter $(addprefix %,$(OSList)),$(basename $(fn))),,$(fn)))
release.HEADERS.COMMON := $(filter-out $(subst _$(_OS),,$(release.HEADERS.OSSPEC)),$(release.HEADERS.COMMON))

# List examples files to populate release/examples.
expat = %.java %.cpp %.h %.txt %.csv
expat += $(if $(OS_is_win),%.bat %.vcxproj %.filters %.user %.sln %makefile_$(_OS),%_$(_OS).lst %makefile_$(_OS) %_$(_OS).sh)
release.EXAMPLES.CPP   := $(filter $(expat),$(shell find examples/cpp  -type f)) $(filter $(expat),$(shell find examples/cpp_sycl -type f))
release.EXAMPLES.DATA  := $(filter $(expat),$(shell find examples/data -type f))
release.EXAMPLES.JAVA  := $(filter $(expat),$(shell find examples/java -type f))

# List env files to populate release.
release.ENV = bin/vars_$(_OS).$(scr)

# List config files to populate release.
release.CONF = bin/config.txt

# List samples files to populate release/examples.
SAMPLES.srcdir:= $(DIR)/samples
spat = %.scala %.java %.cpp %.h %.txt %.csv %.html %.png %.parquet %.blob
spat += $(if $(OS_is_win),%.bat %.vcxproj %.filters %.user %.sln,%_$(_OS).lst %makefile_$(_OS) %.sh)
release.SAMPLES.CPP  := $(if $(wildcard $(SAMPLES.srcdir)/cpp/*),                                                        \
                          $(if $(OS_is_mac),                                                                             \
                            $(filter $(spat),$(shell find $(SAMPLES.srcdir)/cpp -not -wholename '*mpi*' -type f))        \
                          ,                                                                                              \
                            $(filter $(spat),$(shell find $(SAMPLES.srcdir)/cpp -type f))                                \
                          )                                                                                              \
                        )
release.SAMPLES.JAVA := $(if $(wildcard $(SAMPLES.srcdir)/java/*),                                                       \
                          $(if $(or $(OS_is_lnx),$(OS_is_mac),$(OS_is_fbsd)),                                                          \
                            $(filter $(spat),$(shell find $(SAMPLES.srcdir)/java -type f))                               \
                          )                                                                                              \
                        )
release.SAMPLES.SCALA := $(if $(wildcard $(SAMPLES.srcdir)/scala/*),                                                     \
                          $(if $(or $(OS_is_lnx),$(OS_is_mac),$(OS_is_fbsd)),                                                          \
                            $(filter $(spat),$(shell find $(SAMPLES.srcdir)/scala -type f))                              \
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
include makefile.ver
include makefile.lst

THR.srcdir       := $(DIR)/algorithms/threading
CORE.srcdir      := $(DIR)/algorithms/kernel
EXTERNALS.srcdir := $(DIR)/externals

CORE.SERV.srcdir          := $(DIR)/service/kernel
CORE.SERV.COMPILER.srcdir := $(DIR)/service/kernel/compiler/$(CORE.SERV.COMPILER.$(COMPILER))

CORE.srcdirs  := $(CORE.SERV.srcdir) $(CORE.srcdir)                  \
                 $(if $(DAALTHRS),,$(THR.srcdir))                    \
                 $(addprefix $(CORE.SERV.srcdir)/, $(CORE.SERVICES)) \
                 $(addprefix $(CORE.srcdir)/, $(CORE.ALGORITHMS))    \
                 $(CORE.SERV.COMPILER.srcdir) $(EXTERNALS.srcdir)    \
                 $(CORE.SERV.srcdir)/oneapi

CORE.incdirs.rel  := $(addprefix $(RELEASEDIR.include)/,$(addprefix algorithms/,$(CORE.ALGORITHMS.INC)) algorithms data_management/compression data_management/data_source data_management/data services)
CORE.incdirs.thr    := $(THR.srcdir)
CORE.incdirs.core   := $(CORE.SERV.srcdir) $(addprefix $(CORE.SERV.srcdir)/, $(CORE.SERVICES)) $(CORE.srcdir) $(addprefix $(CORE.srcdir)/, $(CORE.ALGORITHMS.FULL)) ## change CORE.ALGORITHMS.FULL --> CORE.ALGORITHMS
CORE.incdirs.common := $(RELEASEDIR.include) $(WORKDIR)
CORE.incdirs.thirdp := $(EXTERNALS.srcdir) $(MKLFPKDIR.include) $(TBBDIR.include)
CORE.incdirs := $(CORE.incdirs.rel) $(CORE.incdirs.thr) $(CORE.incdirs.core) $(CORE.incdirs.common) $(CORE.incdirs.thirdp)

containing = $(foreach v,$2,$(if $(findstring $1,$v),$v))
notcontaining = $(foreach v,$2,$(if $(findstring $1,$v),,$v))
cpy = cp -fp "$<" "$@"

CORE.tmpdir_a := $(WORKDIR)/kernel
CORE.tmpdir_y := $(WORKDIR)/kernel_dll
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
$(WORKDIR.lib)/$(core_y): LOPT += $(if $(OS_is_win),-IMPLIB:$(@:%.dll=%_dll.lib),)
ifdef OS_is_win
$(WORKDIR.lib)/$(core_y:%.dll=%_dll.lib): $(WORKDIR.lib)/$(core_y)
endif
$(CORE.tmpdir_y)/$(core_y:%.$y=%_link.txt): $(CORE.objs_y) $(if $(OS_is_win),$(CORE.tmpdir_y)/dll.res,) | $(CORE.tmpdir_y)/. ; $(WRITE.PREREQS)
$(WORKDIR.lib)/$(core_y):                   $(daaldep.ipp) $(daaldep.vml) $(daaldep.mkl) $(CORE.tmpdir_y)/$(core_y:%.$y=%_link.txt); $(LINK.DYNAMIC) ; $(LINK.DYNAMIC.POST)

$(CORE.objs_a): $(CORE.tmpdir_a)/inc_a_folders.txt
$(CORE.objs_a): COPT += $(-fPIC) $(-cxx11) $(-Zl) $(-DEBC)
$(CORE.objs_a): COPT += -D__TBB_NO_IMPLICIT_LINKAGE -DDAAL_NOTHROW_EXCEPTIONS -DDAAL_HIDE_DEPRECATED
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
$(CORE.objs_y): COPT += -D__DAAL_IMPLEMENTATION -D__TBB_NO_IMPLICIT_LINKAGE -DDAAL_NOTHROW_EXCEPTIONS -DDAAL_HIDE_DEPRECATED $(if $(CHECK_DLL_SIG),-DDAAL_CHECK_DLL_SIG)
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
# Threading parts
#===============================================================================
THR.srcs     := threading.cpp service_thread_pinner.cpp
THR.tmpdir_a := $(WORKDIR)/thread
THR.tmpdir_y := $(WORKDIR)/thread_dll
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

$(THR.objs): COPT += $(-fPIC) $(-cxx11) $(-Zl) $(-DEBC) -DDAAL_HIDE_DEPRECATED
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
JAVA.srcdir      := $(DIR)/lang_interface/java
JAVA.srcdir.full := $(JAVA.srcdir)/com/intel/daal
JAVA.tmpdir      := $(WORKDIR)/java_tmpdir

JNI.srcdir       := $(DIR)/lang_service/java
JNI.srcdir.full  := $(JNI.srcdir)/com/intel/daal
JNI.tmpdir       := $(WORKDIR)/jni_tmpdir

JAVA.srcdirs := $(JAVA.srcdir.full)                                                                                         \
                $(JAVA.srcdir.full)/algorithms $(addprefix $(JAVA.srcdir.full)/algorithms/,$(JJ.ALGORITHMS))                \
                $(JAVA.srcdir.full)/data_management $(addprefix $(JAVA.srcdir.full)/data_management/,$(JJ.DATA_MANAGEMENT)) \
                $(JAVA.srcdir.full)/services $(JAVA.srcdir.full)/utils
JAVA.srcs.f := $(wildcard $(JAVA.srcdirs:%=%/*.java))
JAVA.srcs   := $(subst $(JAVA.srcdir)/,,$(JAVA.srcs.f))

JNI.srcdirs := $(JNI.srcdir.full)                                                                                         \
               $(JNI.srcdir.full)/algorithms $(addprefix $(JNI.srcdir.full)/algorithms/,$(JJ.ALGORITHMS))                 \
               $(JNI.srcdir.full)/data_management $(addprefix $(JNI.srcdir.full)/data_management/,$(JJ.DATA_MANAGEMENT)) \
               $(JNI.srcdir.full)/services
JNI.srcs.f := $(wildcard $(JNI.srcdirs:%=%/*.cpp))
JNI.srcs   := $(subst $(JNI.srcdir)/,,$(JNI.srcs.f))
JNI.objs   := $(addprefix $(JNI.tmpdir)/,$(JNI.srcs:%.cpp=%.$o))

-include $(if $(wildcard $(JNI.tmpdir)/*),$(shell find $(JNI.tmpdir) -name "*.d"))

#----- production of $(daal_jar)
# javac does not generate dependences. Therefore we pass all *.java files to
# a single launch of javac and let it resolve dependences on its own.
# TODO: create hierarchy in java/jni temp folders madually
$(WORKDIR.lib)/$(daal_jar:%.jar=%_link.txt): $(JAVA.srcs.f) | $(WORKDIR.lib)/. ; $(WRITE.PREREQS)
$(WORKDIR.lib)/$(daal_jar):                  $(WORKDIR.lib)/$(daal_jar:%.jar=%_link.txt)
	rm -rf $(JAVA.tmpdir) ; mkdir -p $(JAVA.tmpdir)
	mkdir -p $(JNI.tmpdir)
	javac -classpath $(JAVA.tmpdir) $(-DEBJ) -d $(JAVA.tmpdir) -h $(JNI.tmpdir) @$(WORKDIR.lib)/$(daal_jar:%.jar=%_link.txt)
	jar cvf $@ -C $(JAVA.tmpdir) .

#----- production of JNI dll
$(WORKDIR.lib)/$(jni_so): LOPT += $(-fPIC)
$(WORKDIR.lib)/$(jni_so): LOPT += $(daaldep.rt.thr) $(daaldep.mkl.thr)
$(JNI.tmpdir)/$(jni_so:%.$y=%_link.txt): $(JNI.objs) $(if $(OS_is_win),$(JNI.tmpdir)/dll.res,) $(WORKDIR.lib)/$(core_a) $(WORKDIR.lib)/$(thr_tbb_a) ; $(WRITE.PREREQS)
$(WORKDIR.lib)/$(jni_so):                $(JNI.tmpdir)/$(jni_so:%.$y=%_link.txt); $(LINK.DYNAMIC) ; $(LINK.DYNAMIC.POST)

$(JNI.objs): $(JNI.tmpdir)/inc_j_folders.txt
$(JNI.objs): $(WORKDIR.lib)/$(daal_jar)
$(JNI.objs): COPT += $(-fPIC) $(-cxx11) $(-Zl) $(-DEBC) -DDAAL_NOTHROW_EXCEPTIONS -DDAAL_HIDE_DEPRECATED
$(JNI.objs): COPT += @$(JNI.tmpdir)/inc_j_folders.txt

$(JNI.tmpdir)/inc_j_folders.txt: makefile.lst | $(JNI.tmpdir)/. ; $(call WRITE.PREREQS,$(addprefix -I,$(JNI.tmpdir) $(CORE.incdirs.rel) $(CORE.incdirs.common) $(CORE.incdirs.thirdp) $(JNI.srcdir.full)/include),$(space))

$(JNI.objs): $(JNI.tmpdir)/%.$o: $(JNI.srcdir)/%.cpp; mkdir -p $(@D); $(C.COMPILE)

$(JNI.tmpdir)/dll.res: $(VERSION_DATA_FILE)
$(JNI.tmpdir)/dll.res: RCOPT += -D_DAAL_JAVA_INTERF $(addprefix -I, $(CORE.incdirs.common))
$(JNI.tmpdir)/%.res: %.rc | $(JNI.tmpdir)/. ; $(RC.COMPILE)

#===============================================================================
# Top level targets
#===============================================================================
daal: $(if $(CORE.ALGORITHMS.CUSTOM),                                              \
          _daal _release_c,                                                        \
          _daal _daal_jj _release _release_doc                                     \
      )

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

#-------------------------------------------------------------------------------
# Populating RELEASEDIR
#-------------------------------------------------------------------------------
upd = $(cpy)

_release: info.building.release

#----- releasing static and dynamic libraries
define .release.ay
$3: $2/$1
$(if $(phony-upd),$(eval .PHONY: $2/$1))
$2/$1: $(WORKDIR.lib)/$1 | $2/. ; $(value upd)
endef
$(foreach a,$(release.LIBS_A),$(eval $(call .release.ay,$a,$(RELEASEDIR.libia),_release_c)))
$(foreach y,$(release.LIBS_Y),$(eval $(call .release.ay,$y,$(RELEASEDIR.soia),_release_c)))
$(foreach j,$(release.LIBS_J),$(eval $(call .release.ay,$j,$(RELEASEDIR.soia),_release_jj)))

#----- releasing jar files
_release_jj: $(addprefix $(RELEASEDIR.jardir)/,$(release.JARS))
$(RELEASEDIR.jardir)/%.jar: $(WORKDIR.lib)/%.jar | $(RELEASEDIR.jardir)/. ; $(cpy)

#----- releasing examples
define .release.x
$3: $2/$(subst _$(_OS),,$1)
$2/$(subst _$(_OS),,$1): $(DIR)/$1 | $(dir $2/$1)/.
	$(if $(filter %makefile_win,$1),python ./bin/utils/generate_win_makefile.py $(dir $(DIR)/$1) $(dir $2/$1),$(value cpy))
	$(if $(filter %.sh %.bat,$1),chmod +x $$@)
endef
$(foreach x,$(release.EXAMPLES.DATA),$(eval $(call .release.x,$x,$(RELEASEDIR.daal),_release_common)))
$(foreach x,$(release.EXAMPLES.CPP),$(eval $(call .release.x,$x,$(RELEASEDIR.daal),_release_c)))
$(foreach x,$(release.EXAMPLES.JAVA),$(eval $(call .release.x,$x,$(RELEASEDIR.daal),_release_jj)))

#----- releasing environment scripts
define .release.x
$4: $3/$2
$3/$2: $(DIR)/$1 | $3/. ; $(value cpy)
	$(if $(filter %.sh %.bat,$2),chmod +x $$@)
endef
$(foreach x,$(release.ENV),$(eval $(call .release.x,$x,$(notdir $(subst _$(_OS),,$x)),$(RELEASEDIR.env),_release_common)))
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
$(foreach d,$(release.HEADERS.COMMON),$(eval $(call .release.dd,$d,$(subst include/,$(RELEASEDIR.include)/,$d),_release_c_h)))
$(foreach d,$(release.HEADERS.OSSPEC),$(eval $(call .release.dd,$d,$(subst include/,$(RELEASEDIR.include)/,$(subst _$(_OS),,$d)),_release_c_h)))

#----- releasing static/dynamic Intel(R) TBB libraries
$(RELEASEDIR.tbb.libia) $(RELEASEDIR.tbb.soia): _release_common

define .release.t
_release_common: $2/$(notdir $1)
$2/$(notdir $1): $(call frompf,$1) | $2/. ; $(value cpy)
endef
$(foreach t,$(releasetbb.LIBS_Y),$(eval $(call .release.t,$t,$(RELEASEDIR.tbb.soia))))
$(foreach t,$(releasetbb.LIBS_A),$(eval $(call .release.t,$t,$(RELEASEDIR.tbb.libia))))

#----- releasing MKL GPU FPK libraries

ifneq ($(MKLGPUFPKDIR),)
$(foreach t,$(mklgpufpk.HEADERS),$(eval $(call .release.t,$t,$(RELEASEDIR.include.mklgpufpk))))
$(foreach t,$(mklgpufpk.LIBS_A), $(eval $(call .release.t,$t,$(RELEASEDIR.libia))))
endif

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
endef

daal_dbg:
	@echo "1" "!$(TBBDIR.libia.mac.clang1)!"
	@echo "2" "!$(TBBDIR.libia.mac.clang2)!"
	@echo "3" "!$(TBBDIR.libia.mac.clang22)!"
	@echo "4" "!$(TBBDIR.libia.mac)!"
	@echo "5" ${DYLD_LIBRARY_PATH}
	@echo "6" ${LIBRARY_PATH}
	@echo "7" ${TBBROOT}
