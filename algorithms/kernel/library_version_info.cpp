/* file: library_version_info.cpp */
/*******************************************************************************
* Copyright 2015-2016 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/*
//++
//  Definitions of structures used for environment detection.
//--
*/

#include "library_version_info.h"
#include "_daal_version.h"
#include "service_defines.h"
#include "env_detect.h"

static const char *cpu_long_names[] = {
    "Generic",
    "Intel(R) Supplemental Streaming SIMD Extensions 3",
    "Intel(R) Streaming SIMD Extensions 4.2",
    "Intel(R) Advanced Vector Extensions",
    "Intel(R) Advanced Vector Extensions 2",
    "Intel(R) Advanced Vector Extensions 512 enabled processors (MIC)",
    "Intel(R) Advanced Vector Extensions 512 enabled processors"
};

DAAL_EXPORT daal::services::LibraryVersionInfo::LibraryVersionInfo() :
    majorVersion(MAJORVERSION), minorVersion(MINORVERSION), updateVersion(UPDATEVERSION),
    productStatus(PRODUCTSTATUS), build(BUILD), name(PRODUCT_NAME_STR),
    processor(cpu_long_names[__daal_serv_cpu_detect(daal::services::Environment::avx512_mic |
                                                    daal::services::Environment::avx512)])
{
}

DAAL_EXPORT daal::services::LibraryVersionInfo::~LibraryVersionInfo()
{
}
