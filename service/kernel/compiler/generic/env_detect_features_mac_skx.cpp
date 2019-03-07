/* file: env_detect_features_mac_skx.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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

#if defined(__APPLE__)
#include <sys/sysctl.h>

void __daal_serv_CPUHasAVX512f_enable_it_mac()
{
    int answer = 0;
    size_t answer_size = sizeof(answer);
    ::sysctlbyname("hw.optional.avx512f", &answer, &answer_size, NULL, 0);
    if( answer )
    {
        asm("kandw %k1, %k2, %k3\t");
    }
}
#endif
