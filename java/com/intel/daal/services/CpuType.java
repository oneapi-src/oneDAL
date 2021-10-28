/* file: CpuType.java */
/*******************************************************************************
* Copyright 2014-2021 Intel Corporation
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

/**
 * @ingroup services
 * @{
 */
package com.intel.daal.services;

/**
 * <a name="DAAL-CLASS-SERVICES__CPUTYPE"></a>
 * @brief CPU types
 */
public final class CpuType{

    private int _value;

    /**
     * Constructs the CPU type object using the provided value
     * @param value     Value corresponding to the CPU type object
     */
    public CpuType(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the CPU type object
     * @return Value corresponding to the CPU type object
     */
    public int getValue() {
        return _value;
    }

    private static final int _sse2          = 0;
    private static final int _ssse3         = 1;
    private static final int _sse42         = 2;
    private static final int _avx           = 3;
    private static final int _avx2          = 4;
    private static final int _avx512_mic    = 5;
    private static final int _avx512        = 6;
    private static final int _avx512_mic_e1 = 7;

    public static final CpuType sse2          = new CpuType(_sse2         );  /*!< Intel(R) Streaming SIMD Extensions 2 (Intel(R) SSE2) */
    public static final CpuType ssse3         = new CpuType(_ssse3        );  /*!< Supplemental Streaming SIMD Extensions 3 (SSSE3) */
    public static final CpuType sse42         = new CpuType(_sse42        );  /*!< Intel(R) Streaming SIMD Extensions 4.2 (Intel(R) SSE4.2) */
    public static final CpuType avx           = new CpuType(_avx          );  /*!< Intel(R) Advanced Vector Extensions (Intel(R) AVX) */
    public static final CpuType avx2          = new CpuType(_avx2         );  /*!< Intel(R) Advanced Vector Extensions 2 (Intel(R) AVX2) */
    public static final CpuType avx512_mic    = new CpuType(_avx2   );  /*!< Intel(R) Xeon Phi(TM) processors/coprocessors based on Intel(R) Advanced Vector Extensions 512 (Intel(R) AVX-512) */
    public static final CpuType avx512        = new CpuType(_avx512       );  /*!< Intel(R) Xeon(R) processors based on Intel(R) Advanced Vector Extensions 512 (Intel(R) AVX-512) */
    public static final CpuType avx512_mic_e1 = new CpuType(_avx2);  /*!< Intel(R) Xeon Phi(TM) processors based on Intel(R) Advanced Vector Extensions 512 (Intel(R) AVX-512) with support of AVX512_4FMAPS and AVX512_4VNNIW instruction groups. Should be used as parameter for setCpuId function only. Can`t be received as return value of setCpuId, setCpuId and enableInstructionsSet functions. */
}
/** @} */
