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

MKLFPKDIR:= $(if $(wildcard $(MKLROOT)/include/*),$(subst \,/,$(MKLROOT)),                        \
            $(error Can`t find MKLROOT libs nether in $(DIR)/__deps/mklfpk/$(_OS) not in MKLROOT.))
MKLFPKDIR.include := $(subst \,/,$(MKLFPKDIR)/include)
MKLFPKDIR.libia   := $(subst \,/,$(MKLFPKDIR)/lib)
RELEASEDIR.include.mklgpufpk := $(subst \,/,$(RELEASEDIR.include)/services/internal/sycl/math)

MKLGPUFPKDIR:= $(subst \,/,$(MKLFPKDIR))
MKLGPUFPKDIR.include := $(subst \,/,$(MKLGPUFPKDIR)/include/oneapi)
MKLGPUFPKDIR.lib   := $(subst \,/,$(MKLGPUFPKDIR)/lib)

mklgpufpk.HEADERS := $(subst \,/,$(MKLGPUFPKDIR.include)/mkl.hpp)
mklgpufpk.LIBS_A := $(subst \,/,$(MKLGPUFPKDIR.lib)/$(plib)mkl_sycl$d.$(a))

daaldep.math_backend.incdir := $(subst \,/,$(MKLFPKDIR.include))
daaldep.math_backend_oneapi.incdir := $(subst \,/,$(MKLFPKDIR.include) $(MKLGPUFPKDIR.include))

daaldep.lnx32e.mkl.thr := $(subst \,/,$(MKLFPKDIR.libia)/$(plib)mkl_tbb_thread.$a)
daaldep.lnx32e.mkl.seq := $(subst \,/,$(MKLFPKDIR.libia)/$(plib)mkl_sequential.$a)
daaldep.lnx32e.mkl.core := $(subst \,/,$(MKLFPKDIR.libia)/$(plib)mkl_core.$a $(MKLFPKDIR.libia)/$(plib)mkl_intel_ilp64.$a)
daaldep.lnx32e.mkl.sycl := $(subst \,/,$(MKLGPUFPKDIR.lib)/$(plib)mkl_sycl.$a)

daaldep.win32e.mkl.thr := $(subst \,/,$(MKLFPKDIR.libia)/$(plib)mkl_tbb_thread$d.$a)
daaldep.win32e.mkl.seq := $(subst \,/,$(MKLFPKDIR.libia)/$(plib)mkl_sequential$d.$a)
daaldep.win32e.mkl.core := $(subst \,/,$(MKLFPKDIR.libia)/$(plib)mkl_core$d.$a $(MKLFPKDIR.libia)/$(plib)mkl_intel_ilp64$d.$a)
daaldep.win32e.mkl.sycl := $(subst \,/,$(MKLGPUFPKDIR.lib)/$(plib)mkl_sycl.$d$a)

daaldep.mac32e.mkl.thr := $(subst \,/,$(MKLFPKDIR.libia)/$(plib)daal_mkl_thread.$a)
daaldep.mac32e.mkl.seq := $(subst \,/,$(MKLFPKDIR.libia)/$(plib)daal_mkl_sequential.$a)
daaldep.mac32e.mkl := $(subst \,/,$(MKLFPKDIR.libia)/$(plib)daal_vmlipp_core.$a)

daaldep.fbsd32e.mkl.thr := $(subst \,/,$(MKLFPKDIR.libia)/$(plib)daal_mkl_thread.$a)
daaldep.fbsd32e.mkl.seq := $(subst \,/,$(MKLFPKDIR.libia)/$(plib)daal_mkl_sequential.$a)
daaldep.fbsd32e.mkl := $(subst \,/,$(MKLFPKDIR.libia)/$(plib)daal_vmlipp_core.$a)

daaldep.mkl     := $(subst \,/,$(daaldep.$(PLAT).mkl.core))
daaldep.math_backend.thr := $(subst \,/,$(daaldep.$(PLAT).mkl.thr))
daaldep.math_backend.seq := $(subst \,/,$(daaldep.$(PLAT).mkl.seq))
daaldep.math_backend.sycl := $(subst \,/,$(daaldep.$(PLAT).mkl.sycl))

daaldep.lnx32e.vml :=
daaldep.lnx32e.ipp := $(subst \,/,$(if $(COV.libia),$(COV.libia)/libcov.a))

daaldep.win32e.vml :=
daaldep.win32e.ipp :=

daaldep.mac32e.vml :=
daaldep.mac32e.ipp :=

daaldep.fbsd32e.vml :=
daaldep.fbsd32e.ipp := $(subst \,/,$(if $(COV.libia),$(COV.libia)/libcov.a))

daaldep.vml     := $(subst \,/,$(daaldep.$(PLAT).vml))
daaldep.ipp     := $(subst \,/,$(daaldep.$(PLAT).ipp))

daaldep.math_backend.ext := $(subst \,/,$(daaldep.ipp) $(daaldep.vml) $(daaldep.mkl) $(daaldep.math_backend.thr))
daaldep.math_backend.sycl := $(subst \,/,$(daaldep.math_backend.sycl))
daaldep.math_backend.oneapi := $(subst \,/,$(daaldep.ipp) $(daaldep.vml) $(daaldep.mkl))
