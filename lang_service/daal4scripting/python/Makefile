#*******************************************************************************
# Copyright 2014-2017 Intel Corporation
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
#******************************************************************************/

DAALROOT   ?= please/set/DAALROOT
CNCROOT    ?= please/set/CNCROOT

# Define PREFIX to specify a non-default install directory
PREFIX         ?= $(abspath ./dist)
# Define BUILD_PREFIX if you want intermediate results to be stored elswhere
BUILD_PREFIX   ?= $(abspath ./build)
# Define DAAL4PY_VERSION to set the package version
DAAL4PY_VERSION ?= 0.2018.$(shell date +'%Y%m%d')
# Define SWIG if your swig executable is not on your path (or is not named swig)
SWIG           ?= $(shell which swig)

TOOLSDIR       ?= $(abspath ../common)

LDFLAGS        ?= -ldaal_core -ldaal_thread

PYVER          ?= 3.5

# End of config section
######################

HLAPILANG = Python

all: binpkg

TOOLSDIR_ = $(subst \,/,$(TOOLSDIR))
include $(TOOLSDIR_)/makefile.common

$(BUILD_PREFIX_)/daal.py: $(BUILD_PREFIX_)/hlapi.i typemaps_data.i Makefile numpy.i
	-mkdir -p $(BUILD_PREFIX_)
	$(SWIG) -w401,350,351,394,395,740,742 -python -threads -keyword -interface _daal4py -c++ -o $(BUILD_PREFIX_)/daal_wrap.cpp -I$(abspath ./) -I$(TOOLSDIR_) $(CPPFLAGS) $<

$(PREFIX_)/daal4py-recipe/meta.yaml: meta.yaml build.sh
	-mkdir -p $(PREFIX_)/daal4py-recipe
	sed "s/DAAL4PY_VERSION/$(DAAL4PY_VERSION)/" meta.yaml > $(PREFIX_)/daal4py-recipe/meta.yaml
	cp build.sh $(PREFIX_)/daal4py-recipe

sources: $(BUILD_PREFIX_)/daal.py $(COMMON_SOURCE_FILES)
	-mkdir -p $(PREFIX_)/daal4py/daal4py
	cp $(BUILD_PREFIX_)/daal.py $(PREFIX_)/daal4py/daal4py/__init__.py
	cp $(COMMON_SOURCE_FILES) $(PREFIX_)/daal4py/daal4py/
	sed "s/{{DAAL4PY_VERSION}}/$(DAAL4PY_VERSION)/g" setup.py > $(PREFIX_)/daal4py/setup.py
	cp $(LICENSE) $(PREFIX_)/daal4py/

binpkg: sources $(PREFIX_)/daal4py-recipe/meta.yaml
	cd $(PREFIX_) && env DAAL_INCLUDE=$(DAAL_INCLUDE) conda build --output-folder . -c intel -c intel/label/test --python=$(PYVER) daal4py-recipe

install: binpkg
	conda install $(PREFIX_)/linux-64/$(notdir $(shell cd $(PREFIX_) && conda build -q --output --output-folder . --python=$(PYVER) daal4py-recipe))

build: sources
	cd $(PREFIX_)/daal4py && env DAAL_INCLUDE=$(DAAL_INCLUDE) python setup.py build

setup: sources
	cd $(PREFIX_)/daal4py && env DAAL_INCLUDE=$(DAAL_INCLUDE) python setup.py install
