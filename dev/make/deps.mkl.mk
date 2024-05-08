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

MKLDIR:= $(MKLROOT)
MKLDIR.include := $(MKLDIR)/include
MKLDIR.libia   := $(MKLDIR)/lib

RELEASEDIR.include.mklgpufpk := $(RELEASEDIR.include)/services/internal/sycl/math

# MKLGPUFPKDIR:= $(if $(wildcard $(DIR)/__deps/mklgpufpk/$(_OS)/*),$(DIR)/__deps/mklgpufpk/$(_OS),$(subst \,/,$(MKLGPUFPKROOT)))
# MKLGPUFPKDIR.include := $(MKLROOT)/include

# MKLGPUFPKDIR.libia   := $(MKLROOT)/lib/

mklgpufpk.LIBS_A := $(MKLROOT)/lib/$(plib)mkl_sycl.$a
mklgpufpk.HEADERS :=$(MKLDIR.include)/oneapi/mkl.hpp

daaldep.math_backend.incdir := $(MKLDIR.include)
daaldep.math_backend_oneapi.incdir := $(MKLDIR.include)/oneapi

daaldep.lnx32e.mkl.core := $(MKLROOT)/lib/$(plib)mkl_core.$a $(MKLROOT)/lib/$(plib)mkl_intel_ilp64.$a $(MKLROOT)/lib/$(plib)mkl_tbb_thread.$a
daaldep.lnx32e.mkl.thr := $(MKLROOT)/lib/$(plib)mkl_tbb_thread.$a
daaldep.lnx32e.mkl.seq := $(MKLDIR.libia)/$(plib)mkl_sequential.$a

daaldep.win32e.mkl.iface :=
daaldep.win32e.mkl.core :=
daaldep.win32e.mkl.thr := $(MKLDIR.libia)/daal_mkl_thread$d.$a
daaldep.win32e.mkl.seq := $(MKLDIR.libia)/daal_mkl_sequential.$a
daaldep.win32e.mkl := $(MKLDIR.libia)/$(plib)daal_vmlipp_core$d.$a

daaldep.mac32e.mkl.iface :=
daaldep.mac32e.mkl.core :=
daaldep.mac32e.mkl.thr := $(MKLDIR.libia)/$(plib)daal_mkl_thread.$a
daaldep.mac32e.mkl.seq := $(MKLDIR.libia)/$(plib)daal_mkl_sequential.$a
daaldep.mac32e.mkl := $(MKLDIR.libia)/$(plib)daal_vmlipp_core.$a

daaldep.fbsd32e.mkl.iface :=
daaldep.fbsd32e.mkl.core :=
daaldep.fbsd32e.mkl.thr := $(MKLDIR.libia)/$(plib)daal_mkl_thread.$a
daaldep.fbsd32e.mkl.seq := $(MKLDIR.libia)/$(plib)daal_mkl_sequential.$a
daaldep.fbsd32e.mkl := $(MKLDIR.libia)/$(plib)daal_vmlipp_core.$a


daaldep.mkl     := $(daaldep.$(PLAT).mkl.core)
daaldep.math_backend.thr := $(daaldep.$(PLAT).mkl.thr) $(daaldep.$(PLAT).mkl.core)
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

daaldep.math_backend.ext := $(daaldep.ipp) $(daaldep.vml) $(daaldep.mkl)
