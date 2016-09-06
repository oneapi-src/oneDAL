/* file: CpuTypeEnable.java */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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

package com.intel.daal.services;

/**
 * <a name="DAAL-CLASS-SERVICES__CPUTYPEENABLE"></a>
 * @brief CPU types
 */
public final class CpuTypeEnable{

    private int _value;

    /** Default constructor */
    public CpuTypeEnable(int value) {
        _value = value;
    }

    /**
     * Returns a value corresponding to the identifier of the input object
     * \return Value corresponding to the identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int _cpu_default = 0;
    private static final int _avx512_mic = 1;
    private static final int _avx512 = 2;

    public static final CpuTypeEnable cpu_default = new CpuTypeEnable(_cpu_default);   /*!< Default processor type */
    public static final CpuTypeEnable avx512_mic = new CpuTypeEnable(_avx512_mic);     /*!<
                                                                                        *   Intel(R) Xeon Phi(TM) processors/coprocessors
                                                                                        *   based on Intel(R) Advanced Vector Extensions 512
                                                                                        *   (Intel(R) AVX-512)
                                                                                        */
    public static final CpuTypeEnable avx512 = new CpuTypeEnable(_avx512);             /*!< Intel(R) Xeon(R) processors based on Intel AVX-512 */
}
