#===============================================================================
# Copyright 2017-2019 Intel Corporation
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

##  Content:
##     Intel(R) Data Analytics Acceleration Library samples creation and run
##******************************************************************************

help:
	@echo "Usage: make {libia32|soia32|libintel64|sointel64|help}"
	@echo "[compiler=compiler_name] [mode=mode_name] [sample=sample_name]"
	@echo
	@echo "compiler_name       - intel or gnu"
	@echo "                      Intel(R) C++ Compiler as default"
	@echo
	@echo "mode_name           - can be build or run. Default is run"
	@echo "sample_name         - sample name. Please see daal.lst file"

##------------------------------------------------------------------------------
## samples of using:
##
## make sointel64                - build by Intel(R) C++ Compiler
##                                 and run all samples for Intel(R)64 processor
##                                 family  applications, dynamic linking
##
## make sointel64 mode=build     - build only (not run) by Intel(R) C++ Compiler
##                                 (as default) all samples for Intel(R)64
##                                 processor family  applications, dynamic linking
##
## make help                     - show help
##
##------------------------------------------------------------------------------

include daal.lst

ifndef compiler
    compiler = intel
endif

ifndef DAALROOT
    DAALROOT = ./../..
endif

ifndef TBBROOT
    TBBROOT = ./../../../tbb
endif
TBB_PATH = "$(TBBROOT)/lib/$(_IA)_lin/gcc4.4" "$(TBBROOT)/lib/$(_IA)_lin/gcc4.8"

ifndef sample
    sample = $(KDB)
endif

DAAL_PATH = "$(DAALROOT)/lib/$(_IA)_lin"
DAAL_LIB := $(DAAL_PATH)/libdaal_core.$(RES_EXT) $(DAAL_PATH)/libdaal_thread.$(RES_EXT)

ifneq ($(mode),build)
    override mode = run
endif

EXT_LIB := $(addprefix -L,$(TBB_PATH)) -ltbb -ltbbmalloc -lpthread -ldl $(KDB_LIBRARY_PATH)/c.o

COPTS := -Wall -w -DDAAL_CHECK_PARAMETER -I$(KDB_HEADER_PATH)
LOPTS := -Wl,--start-group $(DAAL_LIB) $(EXT_LIB) -Wl,--end-group

RES_DIR=_results/$(compiler)_$(_IA)_$(RES_EXT)
RES = $(addprefix $(RES_DIR)/, $(if $(filter run, $(mode)), $(addsuffix .res ,$(sample)), $(addsuffix .exe,$(sample))))


ifeq ($(compiler),intel)
    CC = icc
endif

ifeq ($(compiler),gnu)
    CC = g++
    COPTS += $(if $(filter ia32, $(_IA)), -m32, -m64)
endif



.SECONDARY:
$(RES_DIR)/%.exe: ./sources/%.cpp | $(RES_DIR)/.
	$(CC) $(COPTS) $< -o $@ $(LOPTS)

$(RES_DIR)/%.res: $(RES_DIR)/%.exe
	$< > $@

_make_ex: $(RES)

%/.:; mkdir -p $*

libia32:
	$(MAKE) _make_ex _IA=ia32 RES_EXT=a
soia32:
	$(MAKE) _make_ex _IA=ia32 RES_EXT=so
libintel64:
	$(MAKE) _make_ex _IA=intel64 RES_EXT=a
sointel64:
	$(MAKE) _make_ex _IA=intel64 RES_EXT=so
