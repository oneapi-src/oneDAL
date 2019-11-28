/* file: env_detect_features.cpp */
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

/*
//++
//  CPU features detection.
//--
*/

#include <immintrin.h>

#include "env_detect.h"
#include "daal_defines.h"
#include "service_defines.h"
#include "mkl_daal.h"
#include "threading.h"

#include <stdint.h>
#if defined(_MSC_VER)
    #if (_MSC_FULL_VER >= 160040219)
        #include <intrin.h>
    #else
        #error "min VS2010 SP1 compiler is required"
    #endif
#endif

#if defined(__APPLE__)
void __daal_serv_CPUHasAVX512f_enable_it_mac();
#endif

static void run_cpuid(uint32_t eax, uint32_t ecx, uint32_t * abcd)
{
#if defined(_MSC_VER)
    __cpuidex((int *)abcd, eax, ecx);
#else
    uint32_t ebx, edx;
    #if defined(__i386__) && defined(__PIC__)
    /* in case of PIC under 32-bit EBX cannot be clobbered */
    __asm__("movl %%ebx, %%edi \n\t cpuid \n\t xchgl %%ebx, %%edi" : "=D"(ebx), "+a"(eax), "+c"(ecx), "=d"(edx));
    #else
    __asm__("cpuid" : "+b"(ebx), "+a"(eax), "+c"(ecx), "=d"(edx));
    #endif
    abcd[0] = eax;
    abcd[1] = ebx;
    abcd[2] = ecx;
    abcd[3] = edx;
#endif
}

static int check_cpuid(uint32_t eax, uint32_t ecx, int abcd_index, uint32_t mask)
{
    uint32_t abcd[4];

    run_cpuid(eax, ecx, abcd);

    return ((abcd[abcd_index] & mask) == mask);
}

static int check_xgetbv_xcr0_ymm(uint32_t mask)
{
    uint32_t xcr0;
#if defined(_MSC_VER)
    xcr0 = (uint32_t)_xgetbv(0);
#else
    __asm__("xgetbv" : "=a"(xcr0) : "c"(0) : "%edx");
#endif
    return ((xcr0 & mask) == mask); /* checking if xmm and ymm state are enabled in XCR0 */
}

static int check_avx512_features()
{
    /*
    CPUID.(EAX=01H, ECX=0H):ECX.OSXSAVE[bit 27]==1 &&
    CPUID.(EAX=01H, ECX=0H):ECX.AVX    [bit 28]==1
    */
    uint32_t avx_osxsave_mask = ((1 << 27) | (1 << 28));

    /*
    CPUID.(EAX=07H, ECX=0H):EBX.AVX512F [bit 16]==1  &&
    CPUID.(EAX=07H, ECX=0H):EBX.AVX512DQ[bit 17]==1  &&
    CPUID.(EAX=07H, ECX=0H):EBX.AVX512BW[bit 30]==1  &&
    CPUID.(EAX=07H, ECX=0H):EBX.AVX512VL[bit 31]==1
    */
    uint32_t avx512_mask = (1 << 16) | (1 << 17) | (1 << 30) | (1 << 31);

    /*
    E0H - KMASK state, upper 256-bit of ZMM0-ZMM15 and ZMM16-ZMM31 state are enabled by OS
    06H - XMM state and YMM state are enabled by OS
    */
    uint32_t kmask_ymm_mask = 0xE6;

    if (!check_cpuid(1, 0, 2, avx_osxsave_mask))
    {
        return 0;
    }
    if (!check_xgetbv_xcr0_ymm(kmask_ymm_mask))
    {
        return 0;
    }
    if (!check_cpuid(7, 0, 1, avx512_mask))
    {
        return 0;
    }

    return 1;
}

static int check_avx512_mic_features()
{
    /*
    CPUID.(EAX=01H, ECX=0H):ECX.OSXSAVE[bit 27]==1 &&
    CPUID.(EAX=01H, ECX=0H):ECX.AVX    [bit 28]==1
    */
    uint32_t avx_osxsave_mask = ((1 << 27) | (1 << 28));

    /*
    CPUID.(EAX=07H, ECX=0H):EBX.AVX512F [bit 16]==1  &&
    CPUID.(EAX=07H, ECX=0H):EBX.AVX512PF[bit 26]==1  &&
    CPUID.(EAX=07H, ECX=0H):EBX.AVX512ER[bit 27]==1  &&
    CPUID.(EAX=07H, ECX=0H):EBX.AVX512CD[bit 28]==1
    */
    uint32_t avx512_mic_mask = (1 << 16) | (1 << 26) | (1 << 27) | (1 << 28);

    /*
    E0H - KMASK state, upper 256-bit of ZMM0-ZMM15 and ZMM16-ZMM31 state are enabled by OS
    06H - XMM state and YMM state are enabled by OS
    */
    uint32_t kmask_ymm_mask = 0xE6;

    if (!check_cpuid(1, 0, 2, avx_osxsave_mask))
    {
        return 0;
    }
    if (!check_xgetbv_xcr0_ymm(kmask_ymm_mask))
    {
        return 0;
    }
    if (!check_cpuid(7, 0, 1, avx512_mic_mask))
    {
        return 0;
    }

    return 1;
}

static int check_avx2_features()
{
    /* CPUID.(EAX=01H, ECX=0H):ECX.FMA[bit 12]==1     &&
       CPUID.(EAX=01H, ECX=0H):ECX.AES[bit 25]==1     &&
       CPUID.(EAX=01H, ECX=0H):ECX.OSXSAVE[bit 27]==1 &&
       CPUID.(EAX=01H, ECX=0H):ECX.AVX[bit 28]==1 */
    uint32_t fma_aes_osxsave_mask = ((1 << 12) | (1 << 25) | (1 << 27) | (1 << 28));

    /*  CPUID.(EAX=07H, ECX=0H):EBX.AVX2[bit 5]==1  &&
        CPUID.(EAX=07H, ECX=0H):EBX.BMI1[bit 3]==1  &&
        CPUID.(EAX=07H, ECX=0H):EBX.BMI2[bit 8]==1  */
    uint32_t avx2_bmi12_mask = (1 << 5) | (1 << 3) | (1 << 8);

    /* CPUID.(EAX=80000001H):ECX.LZCNT[bit 5]==1 */
    uint32_t lzcnt_mask = (1 << 5);

    if (!check_cpuid(1, 0, 2, fma_aes_osxsave_mask))
    {
        return 0;
    }
    if (!check_xgetbv_xcr0_ymm(6))
    {
        return 0;
    }
    if (!check_cpuid(7, 0, 1, avx2_bmi12_mask))
    {
        return 0;
    }
    if (!check_cpuid(0x80000001, 0, 2, lzcnt_mask))
    {
        return 0;
    }

    return 1;
}

static int check_avx_features()
{
    /* CPUID.(EAX=01H, ECX=0H):ECX.OSXSAVE[bit 27]==1 &&
       CPUID.(EAX=01H, ECX=0H):ECX.AVX[bit 28]==1 */
    uint32_t avx_mask = 0x18000000;

    if (!check_cpuid(1, 0, 2, avx_mask))
    {
        return 0;
    }
    if (!check_xgetbv_xcr0_ymm(6))
    {
        return 0;
    }

    return 1;
}

static int check_sse42_features()
{
    /* CPUID.(EAX=01H, ECX=0H):ECX.SSE4.2[bit 20]==1 */
    uint32_t sse42_mask = 0x100000;

    if (!check_cpuid(1, 0, 2, sse42_mask))
    {
        return 0;
    }

    return 1;
}

static int check_ssse3_features()
{
    /* CPUID.(EAX=01H, ECX=0H):ECX.SSSE3[bit 9]==1 */
    uint32_t ssse3_mask = 0x200;

    if (!check_cpuid(1, 0, 2, ssse3_mask))
    {
        return 0;
    }

    return 1;
}

int __daal_serv_cpu_detect(int enable)
{
    if ((enable & daal::services::Environment::avx512_mic_e1) == daal::services::Environment::avx512_mic_e1)
    {
        fpk_serv_enable_instructions(MKL_ENABLE_AVX512_MIC_E1);
    }

#if defined(__APPLE__)
    __daal_serv_CPUHasAVX512f_enable_it_mac();
#endif
    if (check_avx512_features())
    {
        return daal::avx512;
    }

    if (check_avx512_mic_features())
    {
        return daal::avx512_mic;
    }

    if (check_avx2_features())
    {
        return daal::avx2;
    }

    if (check_avx_features())
    {
        return daal::avx;
    }

    if (check_sse42_features())
    {
        return daal::sse42;
    }

    if (check_ssse3_features())
    {
        return daal::ssse3;
    }

    return daal::sse2;
}
