/* file: env_detect_features_mac_skx.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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

#if defined(__APPLE__)
    #include <sys/sysctl.h>

void __daal_serv_CPUHasAVX512f_enable_it_mac()
{
    int answer         = 0;
    size_t answer_size = sizeof(answer);
    ::sysctlbyname("hw.optional.avx512f", &answer, &answer_size, NULL, 0);
    if (answer)
    {
        asm("kandw %k1, %k2, %k3\t");
    }
}
#endif
