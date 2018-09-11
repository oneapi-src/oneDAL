/* file: library_version_info.cpp */
/*******************************************************************************
* Copyright 2015-2018 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
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
#include "mkl_daal.h"

static const char *cpu_long_names[] = {
    "Generic",
    "Supplemental Streaming SIMD Extensions 3",
    "Intel(R) Streaming SIMD Extensions 4.2",
    "Intel(R) Advanced Vector Extensions",
    "Intel(R) Advanced Vector Extensions 2",
    "Intel(R) Xeon Phi(TM) processors/coprocessors based on Intel(R) Advanced Vector Extensions 512",
    "Intel(R) Xeon(R) processors based on Intel(R) Advanced Vector Extensions 512",
    "Intel(R) Xeon Phi(TM) processors based on Intel(R) Advanced Vector Extensions 512 with support of AVX512_4FMAPS and AVX512_4VNNIW instruction groups"
};

DAAL_EXPORT daal::services::LibraryVersionInfo::LibraryVersionInfo() :
    majorVersion(MAJORVERSION), minorVersion(MINORVERSION), updateVersion(UPDATEVERSION),
    productStatus(PRODUCTSTATUS), build(BUILD), build_rev(BUILD_REV), name(PRODUCT_NAME_STR),
    processor(cpu_long_names[daal::services::Environment::getInstance()->getCpuId()+2*fpk_serv_cpuisknm()])
{
}

DAAL_EXPORT daal::services::LibraryVersionInfo::~LibraryVersionInfo()
{
}
