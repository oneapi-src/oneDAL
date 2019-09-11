#!/bin/tcsh
#===============================================================================
# Copyright 2014-2019 Intel Corporation.
#
# This software and the related documents are Intel copyrighted  materials,  and
# your use of  them is  governed by the  express license  under which  they were
# provided to you (License).  Unless the License provides otherwise, you may not
# use, modify, copy, publish, distribute,  disclose or transmit this software or
# the related documents without Intel's prior written permission.
#
# This software and the related documents  are provided as  is,  with no express
# or implied  warranties,  other  than those  that are  expressly stated  in the
# License.
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
