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

set __daal_tmp_script_name="daalvars.csh"
set __daal_tmp_target_arch=""

if ( $#argv == 0 ) then
    if ( $?COMPILERVARS_ARCHITECTURE ) then
        set __daal_tmp_target_arch="$COMPILERVARS_ARCHITECTURE"
    endif
    if ( $?DAALVARS_ARCHITECTURE ) then
        set __daal_tmp_target_arch="$DAALVARS_ARCHITECTURE"
    endif
else
    while ( "$1" != "" )
        if      ( "$1" == "ia32"      ) then
            set __daal_tmp_target_arch="ia32"
        else if ( "$1" == "intel64"   ) then
            set __daal_tmp_target_arch="intel64"
        else
            echo ""
            echo "ERROR: Unknown option '$1'"
            goto Help
        endif
        shift
    end
endif

if ( "${__daal_tmp_target_arch}" != "intel64" ) then
    if ( "${__daal_tmp_target_arch}" != "ia32" ) then
        echo "ERROR: architecture is not defined. Accepted values: ia32, intel64"
        goto Help
    endif
endif

if ( ${?CPATH} ) then
    setenv CPATH "${DAALROOT}/include:${CPATH}"
else
    setenv CPATH "${DAALROOT}/include"
endif

if ( ${?LIBRARY_PATH} ) then
    if ( ${?TBBROOT} ) then
        setenv LIBRARY_PATH "${DAALROOT}/lib/${__daal_tmp_target_arch}_lin:${LIBRARY_PATH}"
    else
        setenv LIBRARY_PATH "${DAALROOT}/lib/${__daal_tmp_target_arch}_lin:${DAALROOT}/../tbb/lib/${__daal_tmp_target_arch}_lin/gcc4.4:${LIBRARY_PATH}"
    endif
else
    if ( ${?TBBROOT} ) then
        setenv LIBRARY_PATH "${DAALROOT}/lib/${__daal_tmp_target_arch}_lin"
    else
        setenv LIBRARY_PATH "${DAALROOT}/lib/${__daal_tmp_target_arch}_lin:${DAALROOT}/../tbb/lib/${__daal_tmp_target_arch}_lin/gcc4.4"
    endif
endif

if ( ${?LD_LIBRARY_PATH} ) then
    if ( ${?TBBROOT} ) then
        setenv LD_LIBRARY_PATH "${DAALROOT}/lib/${__daal_tmp_target_arch}_lin:${LD_LIBRARY_PATH}"
    else
        setenv LD_LIBRARY_PATH "${DAALROOT}/lib/${__daal_tmp_target_arch}_lin:${DAALROOT}/../tbb/lib/${__daal_tmp_target_arch}_lin/gcc4.4:${LD_LIBRARY_PATH}"
    endif
else
    if ( ${?TBBROOT} ) then
        setenv LD_LIBRARY_PATH "${DAALROOT}/lib/${__daal_tmp_target_arch}_lin"
    else
        setenv LD_LIBRARY_PATH "${DAALROOT}/lib/${__daal_tmp_target_arch}_lin:${DAALROOT}/../tbb/lib/${__daal_tmp_target_arch}_lin/gcc4.4"
    endif
endif

if ( ${?CLASSPATH} ) then
    setenv CLASSPATH "${DAALROOT}/lib/daal.jar:${CLASSPATH}"
else
    setenv CLASSPATH "${DAALROOT}/lib/daal.jar"
endif

goto End

Help:
    echo "Syntax: source $__daal_tmp_script_name <arch>"
    echo "Where <arch> is one of:"
    echo "  ia32      - setup environment for IA-32 architecture"
    echo "  intel64   - setup environment for Intel(R) 64 architecture"
    echo ""
    echo "If the arguments to the sourced script are ignored (consult docs for"
    echo "your shell) the alternative way to specify target is environment"
    echo "variables COMPILERVARS_ARCHITECTURE or DAALVARS_ARCHITECTURE to pass"
    echo "<arch> to the script."
    echo ""
    exit 1;

End: # Clean up of internal settings
    unset __daal_tmp_target_arch
    unset __daal_tmp_cpro_path
    unset __daal_tmp_script_name
