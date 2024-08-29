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

MKLFPKDIR:= $(subst \,/,$(MKLROOT))
MKLFPKDIR.include := $(MKLFPKDIR)/include
MKLFPKDIR.libia   := $(MKLFPKDIR)/lib
RELEASEDIR.include.mklgpufpk := $(RELEASEDIR.include)/services/internal/sycl/math

MKLGPUFPKDIR:= $(MKLFPKDIR)
MKLGPUFPKDIR.include := $(MKLGPUFPKDIR)/include/oneapi
MKLGPUFPKDIR.lib   := $(MKLGPUFPKDIR)/lib

mklgpufpk.HEADERS := $(MKLGPUFPKDIR.include)/mkl.hpp
mklgpufpk.LIBS_A := $(MKLGPUFPKDIR.lib)/$(plib)mkl_sycl$d.$a

daaldep.math_backend.incdir := $(MKLFPKDIR.include)
daaldep.math_backend_oneapi.incdir := $(MKLFPKDIR.include) $(MKLGPUFPKDIR.include)

daaldep.lnx32e.mkl.thr := $(MKLFPKDIR.libia)/$(plib)mkl_tbb_thread.$a
daaldep.lnx32e.mkl.seq := $(MKLFPKDIR.libia)/$(plib)mkl_sequential.$a
daaldep.lnx32e.mkl.core := $(MKLFPKDIR.libia)/$(plib)mkl_core.$a $(MKLFPKDIR.libia)/$(plib)mkl_intel_ilp64.$a
daaldep.lnx32e.mkl.sycl := $(MKLGPUFPKDIR.lib)/$(plib)mkl_sycl.$a

daaldep.win32e.mkl.thr := $(MKLFPKDIR.libia)/mkl_tbb_thread$d.$a
daaldep.win32e.mkl.seq := $(MKLFPKDIR.libia)/mkl_sequential.$a
mkl_core_lib := $(MKLFPKDIR.libia)/mkl_core$d.$a
mkl_intel_ilp64_lib := $(MKLFPKDIR.libia)/mkl_intel_ilp64$d.$a
daaldep.win32e.mkl.core := $(mkl_core_lib) $(mkl_intel_ilp64_lib)
daaldep.win32e.mkl.sycl := $(MKLGPUFPKDIR.lib)/mkl_sycl.$d$a

daaldep.mac32e.mkl.thr := $(MKLFPKDIR.libia)/$(plib)daal_mkl_thread.$a
daaldep.mac32e.mkl.seq := $(MKLFPKDIR.libia)/$(plib)daal_mkl_sequential.$a
daaldep.mac32e.mkl := $(MKLFPKDIR.libia)/$(plib)daal_vmlipp_core.$a

daaldep.fbsd32e.mkl.thr := $(MKLFPKDIR.libia)/$(plib)daal_mkl_thread.$a
daaldep.fbsd32e.mkl.seq := $(MKLFPKDIR.libia)/$(plib)daal_mkl_sequential.$a
daaldep.fbsd32e.mkl := $(MKLFPKDIR.libia)/$(plib)daal_vmlipp_core.$a

daaldep.mkl     := $(daaldep.$(PLAT).mkl.core)
daaldep.math_backend.thr := $(daaldep.$(PLAT).mkl.thr)
daaldep.math_backend.seq := $(daaldep.$(PLAT).mkl.seq)
daaldep.math_backend.sycl := $(daaldep.$(PLAT).mkl.sycl)

daaldep.lnx32e.vml :=
daaldep.lnx32e.ipp := $(if $(COV.libia),$(COV.libia)/libcov.a)

daaldep.win32e.vml :=
daaldep.win32e.ipp :=

daaldep.mac32e.vml :=
daaldep.mac32e.ipp :=

daaldep.fbsd32e.vml :=
daaldep.fbsd32e.ipp := (if $(COV.libia),$(COV.libia)/libcov.a)

daaldep.vml     := $(daaldep.$(PLAT).vml)
daaldep.ipp     := $(daaldep.$(PLAT).ipp)

daaldep.math_backend.ext := $(daaldep.ipp) $(daaldep.vml) $(daaldep.mkl) $(daaldep.math_backend.thr)
daaldep.math_backend.sycl := $(daaldep.math_backend.sycl)
daaldep.math_backend.oneapi := $(daaldep.ipp) $(daaldep.vml) $(daaldep.mkl)
