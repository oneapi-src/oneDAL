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
MKLFPKDIR.include := $(MKLFPKDIR)/include $(MKLFPKDIR)/$(if $(OS_is_fbsd),lnx,$(_OS))/include
MKLFPKDIR.libia   := $(MKLFPKDIR)/$(if $(OS_is_fbsd),lnx,$(_OS))/lib/$(_IA)

RELEASEDIR.include.mklgpufpk := $(RELEASEDIR.include)/services/internal/sycl/math

MKLGPUFPKDIR:= $(if $(wildcard $(DIR)/__deps/mklgpufpk/$(_OS)/*),$(DIR)/__deps/mklgpufpk/$(_OS),$(subst \,/,$(MKLGPUFPKROOT)))
MKLGPUFPKDIR.include := $(MKLGPUFPKDIR)/include
MKLGPUFPKDIR.lib   := $(MKLGPUFPKDIR)/lib/

mklgpufpk.LIBS_A := $(MKLGPUFPKDIR.lib)/$(plib)daal_sycl$d.$(a)
mklgpufpk.HEADERS := $(MKLGPUFPKDIR.include)/mkl_dal_sycl.hpp $(MKLGPUFPKDIR.include)/mkl_dal_blas_sycl.hpp

daaldep.math_backend.incdir := $(MKLFPKDIR.include) $(MKLGPUFPKDIR.include)
daaldep.math_backend_oneapi.incdir := $(MKLFPKDIR.include) $(MKLGPUFPKDIR.include)

daaldep.lnxx86_64.mkl.thr := $(MKLFPKDIR.libia)/$(plib)daal_mkl_thread.$a
daaldep.lnxx86_64.mkl.seq := $(MKLFPKDIR.libia)/$(plib)daal_mkl_sequential.$a
daaldep.lnxx86_64.mkl := $(MKLFPKDIR.libia)/$(plib)daal_vmlipp_core.$a

daaldep.winx86_64.mkl.thr := $(MKLFPKDIR.libia)/daal_mkl_thread$d.$a
daaldep.winx86_64.mkl.seq := $(MKLFPKDIR.libia)/daal_mkl_sequential.$a
daaldep.winx86_64.mkl := $(MKLFPKDIR.libia)/$(plib)daal_vmlipp_core$d.$a

daaldep.macx86_64.mkl.thr := $(MKLFPKDIR.libia)/$(plib)daal_mkl_thread.$a
daaldep.macx86_64.mkl.seq := $(MKLFPKDIR.libia)/$(plib)daal_mkl_sequential.$a
daaldep.macx86_64.mkl := $(MKLFPKDIR.libia)/$(plib)daal_vmlipp_core.$a

daaldep.fbsdx86_64.mkl.thr := $(MKLFPKDIR.libia)/$(plib)daal_mkl_thread.$a
daaldep.fbsdx86_64.mkl.seq := $(MKLFPKDIR.libia)/$(plib)daal_mkl_sequential.$a
daaldep.fbsdx86_64.mkl := $(MKLFPKDIR.libia)/$(plib)daal_vmlipp_core.$a


daaldep.mkl     := $(daaldep.$(PLAT).mkl)
daaldep.math_backend.thr := $(daaldep.$(PLAT).mkl.thr)
daaldep.math_backend.seq := $(daaldep.$(PLAT).mkl.seq) $(daaldep.mkl)

daaldep.lnxx86_64.vml :=
daaldep.lnxx86_64.ipp := $(if $(COV.libia),$(COV.libia)/libcov.a)

daaldep.winx86_64.vml :=
daaldep.winx86_64.ipp :=

daaldep.macx86_64.vml :=
daaldep.macx86_64.ipp :=

daaldep.fbsdx86_64.vml :=
daaldep.fbsdx86_64.ipp := $(if $(COV.libia),$(COV.libia)/libcov.a)

daaldep.vml     := $(daaldep.$(PLAT).vml)
daaldep.ipp     := $(daaldep.$(PLAT).ipp)

daaldep.math_backend.ext := $(daaldep.ipp) $(daaldep.vml) $(daaldep.mkl)
