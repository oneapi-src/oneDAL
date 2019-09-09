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

export PREFIX = ./dist
# Define BUILD_PREFIX if you want intermediate results to be stored elswhere
export BUILD_PREFIX = ./build
# Define PYDAAL_VERSION to set the pydaal version
export PYDAAL_VERSION = 0.0.0dev$(shell date +'%Y%m%d')
# DEFINE DOXYFILE to set the doxyfile to be used for doc generation
# use doxygen_conf_python.txt or doxygen_conf_python_web.txt
export DOXYFILE = doxygen_conf_python.txt
# set to NO to create long file names with doxygen
export DOXY_SHORT = YES

ALGOS ?= $(patsubst %.i.tmpl,%,$(wildcard algorithm*.i.tmpl))
DMGNT =  $(patsubst %.i.tmpl,%,$(wildcard data_management*.i.tmpl))

IF_CONFIGS = $(sort $(DMGNT)) $(sort daal services $(ALGOS))

# We want to detect new/removed interface files.
# We cannot create arbitrarily long file-names.
# Hence we generate a hash id for the list of interface files
IF_HASH = if_hash.$(shell python -c 'import hashlib; print(hashlib.md5("$(IF_CONFIGS)".encode("utf-8")).hexdigest())')
$(BUILD_PREFIX)/$(IF_HASH):
	-mkdir -p $(BUILD_PREFIX)
	touch $@

comma := ,
$(BUILD_PREFIX)/all_namespaces.j2: Makefile $(BUILD_PREFIX)/$(IF_HASH)
	-mkdir -p $(BUILD_PREFIX)
	@echo Generating list of namespaces...
	@echo '{% set all_namespaces = [' > $@.new
	@echo $(patsubst %," '%'$(comma)",$(IF_CONFIGS)) >> $@.new
	@echo '] %}' >> $@.new
	@echo done
	if [ -f $@ ] && cmp -s $@ $@.new ; then : ; else cp $@.new $@ ; fi

$(BUILD_PREFIX)/make.rules: make.rules.tmpl Makefile jinjadefs.tmpl jinja.py $(BUILD_PREFIX)/all_namespaces.j2
	-mkdir -p $(BUILD_PREFIX)
	cp $(BUILD_PREFIX)/all_namespaces.j2 .
	python ./jinja.py --vars="BUILD_PREFIX=$(BUILD_PREFIX)" $< > $@
	rm all_namespaces.j2

-include $(BUILD_PREFIX)/make.rules

touchbuild:
	touch $(BUILD_PREFIX)/*.h
	touch $(BUILD_PREFIX)/*/*.i
	touch $(BUILD_PREFIX)/*/*.h
	touch $(BUILD_PREFIX)/*/*.cpp
	touch $(BUILD_PREFIX)/*/*.o
	touch $(BUILD_PREFIX)/*/*.so

ifneq ($(strip $(_THESE_)),)
these:
	$(MAKE) apiifiles
	$(MAKE) $(BUILD_PREFIX)/api/sp_list.i
	- rm $(BUILD_PREFIX)/*/*$(_THESE_)*.*
	- $(MAKE) touchbuild
	$(MAKE) daal
endif
