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
#  Math backend (MKL) definitions for makefile
#--
MKLFPKDIR:= $(if $(wildcard $(DIR)/__deps/mklfpk/$(_OS)/*),$(DIR)/__deps/mklfpk,                        \
            $(if $(wildcard $(MKLROOT)/include/*),$(subst \,/,$(MKLROOT)),                        \
            $(error Can`t find MKLFPK libs nether in $(DIR)/__deps/mklfpk/$(_OS) not in MKLFPKROOT.)))
MKLFPKDIR.include := $(MKLFPKDIR)/include $(MKLFPKDIR)/$(if $(OS_is_fbsd),lnx,$(_OS))/include
MKLFPKDIR.libia   := $(MKLFPKDIR)/$(if $(OS_is_fbsd),lnx,$(_OS))/lib
RELEASEDIR.include.mklgpufpk := $(RELEASEDIR.include)/services/internal/sycl/math

MKLGPUFPKDIR:= $(if $(wildcard $(DIR)/__deps/mklgpufpk/$(_OS)/*),$(DIR)/__deps/mklgpufpk/$(_OS),$(subst \,/,$(MKLROOT)))
MKLGPUFPKDIR.include := $(MKLGPUFPKDIR)/include/oneapi
MKLGPUFPKDIR.lib   := $(MKLGPUFPKDIR)/lib

mklgpufpk.HEADERS := $(MKLGPUFPKDIR.include)/mkl.hpp
mklgpufpk.LIBS_A := $(MKLGPUFPKDIR.lib)/$(plib)mkl_sycl$d.$(a)

daaldep.math_backend.incdir := $(MKLFPKDIR.include)
daaldep.math_backend_oneapi.incdir := $(MKLFPKDIR.include) $(MKLGPUFPKDIR.include)

daaldep.lnx32e.mkl.thr := $(MKLROOT)/lib/$(plib)mkl_tbb_thread.$a
daaldep.lnx32e.mkl.seq := $(MKLROOT)/lib/$(plib)mkl_sequential.$a
daaldep.lnx32e.mkl.core := $(MKLROOT)/lib/$(plib)mkl_core.$a $(MKLROOT)/lib/$(plib)mkl_intel_ilp64.$a

daaldep.win32e.mkl.thr := $(MKLROOT)/lib/$(plib)mkl_tbb_thread.$a
daaldep.win32e.mkl.seq := $(MKLROOT)/lib/$(plib)mkl_sequential.$a
daaldep.win32e.mkl.core := $(MKLROOT)/lib/$(plib)mkl_core$d.$a $(MKLROOT)/lib/$(plib)mkl_intel_ilp64$d.$a

daaldep.mac32e.mkl.thr := $(MKLFPKDIR.libia)/$(plib)daal_mkl_thread.$a
daaldep.mac32e.mkl.seq := $(MKLFPKDIR.libia)/$(plib)daal_mkl_sequential.$a
daaldep.mac32e.mkl := $(MKLFPKDIR.libia)/$(plib)daal_vmlipp_core.$a

daaldep.fbsd32e.mkl.thr := $(MKLFPKDIR.libia)/$(plib)daal_mkl_thread.$a
daaldep.fbsd32e.mkl.seq := $(MKLFPKDIR.libia)/$(plib)daal_mkl_sequential.$a
daaldep.fbsd32e.mkl := $(MKLFPKDIR.libia)/$(plib)daal_vmlipp_core.$a


daaldep.mkl     := $(daaldep.$(PLAT).mkl.core)
daaldep.math_backend.thr := $(daaldep.$(PLAT).mkl.thr)
daaldep.math_backend.seq := $(daaldep.$(PLAT).mkl.seq)

daaldep.lnx32e.vml :=
daaldep.lnx32e.ipp := $(if $(COV.libia),$(COV.libia)/libcov.a)

daaldep.win32e.vml :=
daaldep.win32e.ipp :=

daaldep.mac32e.vml :=
daaldep.mac32e.ipp :=

daaldep.fbsd32e.vml :=
daaldep.fbsd32e.ipp := $(if $(COV.libia),$(COV.libia)/libcov.a)

daaldep.vml     := $(daaldep.$(PLAT).vml)
daaldep.ipp     := $(daaldep.$(PLAT).ipp)

daaldep.math_backend.ext := $(daaldep.ipp) $(daaldep.vml) $(daaldep.mkl) $(daaldep.math_backend.thr)
daaldep.math_backend.sycl := $(daaldep.ipp) $(daaldep.vml) $(daaldep.mkl) $(daaldep.math_backend.thr)
daaldep.math_backend.oneapi := $(daaldep.ipp) $(daaldep.vml) $(daaldep.mkl)
