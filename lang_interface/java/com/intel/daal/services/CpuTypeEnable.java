/* file: CpuTypeEnable.java */
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
 * <a name="DAAL-CLASS-SERVICES__CPUTYPEENABLE"></a>
 * @brief CPU types
 */
public final class CpuTypeEnable{

    private int _value;

    /**
     * Constructs the CPU type object using the provided value
     * @param value     Value corresponding to the CPU type object
     */
    public CpuTypeEnable(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the CPU type object
     * @return Value corresponding to the CPU type object
     */
    public int getValue() {
        return _value;
    }

    private static final int _cpu_default = 0;
    private static final int _avx512_mic = 1;
    private static final int _avx512 = 2;
    private static final int _avx512_mic_e1 = 4;

    public static final CpuTypeEnable cpu_default = new CpuTypeEnable(_cpu_default);       /*!< Default processor type */
    public static final CpuTypeEnable avx512_mic = new CpuTypeEnable(_avx512_mic);         /*!< Intel(R) Xeon Phi(TM) processors/coprocessors based on Intel(R) Advanced Vector Extensions 512 (Intel(R) AVX-512) */
    public static final CpuTypeEnable avx512 = new CpuTypeEnable(_avx512);                 /*!< Intel(R) Xeon(R) processors based on Intel(R) Advanced Vector Extensions 512 (Intel(R) AVX-512) */
    public static final CpuTypeEnable avx512_mic_e1 = new CpuTypeEnable(_avx512_mic_e1);   /*!< Intel(R) Xeon Phi(TM) processors based on Intel(R) Advanced Vector Extensions 512 (Intel(R) AVX-512) with support of AVX512_4FMAPS and AVX512_4VNNIW instruction groups */
}
/** @} */
