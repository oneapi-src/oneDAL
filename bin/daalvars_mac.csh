#!/bin/tcsh
#===============================================================================
# Copyright 2014-2016 Intel Corporation
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

set __daal_tmp_cpro_path=<INSTALLDIR>
setenv DAALROOT ${__daal_tmp_cpro_path}/daal

if ( ${?CPATH} ) then
    setenv CPATH "${DAALROOT}/include:${CPATH}"
else
    setenv CPATH "${DAALROOT}/include"
endif

if ( ${?LIBRARY_PATH} ) then
    if ( ${?TBBROOT} ) then
        setenv LIBRARY_PATH "${DAALROOT}/lib:${LIBRARY_PATH}"
    else
        setenv LIBRARY_PATH "${DAALROOT}/lib:${DAALROOT}/../tbb/lib:${LIBRARY_PATH}"
    endif
else
    if ( ${?TBBROOT} ) then
        setenv LIBRARY_PATH "${DAALROOT}/lib"
    else
        setenv LIBRARY_PATH "${DAALROOT}/lib:${DAALROOT}/../tbb/lib"
    endif
endif

if ( ${?DYLD_LIBRARY_PATH} ) then
    if ( ${?TBBROOT} ) then
        setenv DYLD_LIBRARY_PATH "${DAALROOT}/lib:${DYLD_LIBRARY_PATH}"
    else
        setenv DYLD_LIBRARY_PATH "${DAALROOT}/lib:${DAALROOT}/../tbb/lib:${DYLD_LIBRARY_PATH}"
    endif
else
    if ( ${?TBBROOT} ) then
        setenv DYLD_LIBRARY_PATH "${DAALROOT}/lib"
    else
        setenv DYLD_LIBRARY_PATH "${DAALROOT}/lib:${DAALROOT}/../tbb/lib"
    endif
endif

if ( ${?CLASSPATH} ) then
    setenv CLASSPATH "${DAALROOT}/lib/daal.jar:${CLASSPATH}"
else
    setenv CLASSPATH "${DAALROOT}/lib/daal.jar"
endif

# Clean up of internal settings
unset __daal_tmp_cpro_path
