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
            $(if $(wildcard $(MKLFPKROOT)/include/*),$(subst \,/,$(MKLFPKROOT)),                        \
            $(error Can`t find MKLFPK libs nether in $(DIR)/__deps/mklfpk/$(_OS) not in MKLFPKROOT.)))
MKLFPKDIR.include := $(MKLFPKDIR)/include $(MKLFPKDIR)/$(if $(OS_is_fbsd),lnx,$(_OS))/include /opt/intel/oneapi/mkl/latest/include
MKLFPKDIR.libia   := $(MKLFPKDIR)/$(if $(OS_is_fbsd),lnx,$(_OS))/lib/$(_IA)

RELEASEDIR.include.mklgpufpk := $(RELEASEDIR.include)/services/internal/sycl/math

MKLGPUFPKDIR:= $(if $(wildcard $(DIR)/__deps/mklgpufpk/$(_OS)/*),$(DIR)/__deps/mklgpufpk/$(_OS),$(subst \,/,$(MKLGPUFPKROOT)))
MKLGPUFPKDIR.include := $(MKLGPUFPKDIR)/include
ifeq ($(MKL_FPK_GPU_VERSION_LINE),2024.0.0)
    MKLGPUFPKDIR.libia   := $(MKLGPUFPKDIR)/lib/
else
    MKLGPUFPKDIR.libia   := $(MKLGPUFPKDIR)/lib/$(_IA)
endif
mklgpufpk.LIBS_A := /opt/intel/oneapi/mkl/latest/lib/$(_IA)/$(plib)mkl_sycl.$a
mklgpufpk.HEADERS :=

daaldep.math_backend.incdir := $(MKLFPKDIR.include) $(MKLGPUFPKDIR.include)
daaldep.math_backend_oneapi.incdir := $(MKLFPKDIR.include) $(MKLGPUFPKDIR.include)

daaldep.lnx32e.mkl.core := /opt/intel/oneapi/mkl/latest/lib/$(_IA)/$(plib)mkl_core.$a
daaldep.lnx32e.mkl.iface := /opt/intel/oneapi/mkl/latest/lib/$(_IA)/$(plib)mkl_intel_ilp64.$a
daaldep.lnx32e.mkl.thr := /opt/intel/oneapi/mkl/latest/lib/$(_IA)/$(plib)mkl_tbb_thread.$a
daaldep.lnx32e.mkl.seq := $(MKLFPKDIR.libia)/$(plib)mkl_sequential.$a
daaldep.lnx32e.mkl := $(MKLFPKDIR.libia)/$(plib)daal_vmlipp_core.$a

daaldep.win32e.mkl.iface :=
daaldep.win32e.mkl.core :=
daaldep.win32e.mkl.thr := $(MKLFPKDIR.libia)/daal_mkl_thread$d.$a
daaldep.win32e.mkl.seq := $(MKLFPKDIR.libia)/daal_mkl_sequential.$a
daaldep.win32e.mkl := $(MKLFPKDIR.libia)/$(plib)daal_vmlipp_core$d.$a

daaldep.mac32e.mkl.iface :=
daaldep.mac32e.mkl.core :=
daaldep.mac32e.mkl.thr := $(MKLFPKDIR.libia)/$(plib)daal_mkl_thread.$a
daaldep.mac32e.mkl.seq := $(MKLFPKDIR.libia)/$(plib)daal_mkl_sequential.$a
daaldep.mac32e.mkl := $(MKLFPKDIR.libia)/$(plib)daal_vmlipp_core.$a

daaldep.fbsd32e.mkl.iface :=
daaldep.fbsd32e.mkl.core :=
daaldep.fbsd32e.mkl.thr := $(MKLFPKDIR.libia)/$(plib)daal_mkl_thread.$a
daaldep.fbsd32e.mkl.seq := $(MKLFPKDIR.libia)/$(plib)daal_mkl_sequential.$a
daaldep.fbsd32e.mkl := $(MKLFPKDIR.libia)/$(plib)daal_vmlipp_core.$a


daaldep.mkl     := $(daaldep.$(PLAT).mkl)
daaldep.math_backend.thr := $(daaldep.$(PLAT).mkl.iface) $(daaldep.$(PLAT).mkl.thr) $(daaldep.$(PLAT).mkl.core)
daaldep.math_backend.seq := $(daaldep.$(PLAT).mkl.seq) $(daaldep.mkl)

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
