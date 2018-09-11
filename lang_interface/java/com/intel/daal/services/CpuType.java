/* file: CpuType.java */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
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
    public static final CpuType avx512_mic    = new CpuType(_avx512_mic   );  /*!< Intel(R) Xeon Phi(TM) processors/coprocessors based on Intel(R) Advanced Vector Extensions 512 (Intel(R) AVX-512) */
    public static final CpuType avx512        = new CpuType(_avx512       );  /*!< Intel(R) Xeon(R) processors based on Intel(R) Advanced Vector Extensions 512 (Intel(R) AVX-512) */
    public static final CpuType avx512_mic_e1 = new CpuType(_avx512_mic_e1);  /*!< Intel(R) Xeon Phi(TM) processors based on Intel(R) Advanced Vector Extensions 512 (Intel(R) AVX-512) with support of AVX512_4FMAPS and AVX512_4VNNIW instruction groups. Should be used as parameter for setCpuId function only. Can`t be received as return value of setCpuId, setCpuId and enableInstructionsSet functions. */
}
/** @} */
