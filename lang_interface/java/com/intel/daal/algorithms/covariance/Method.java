/* file: Method.java */
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

/**
 * @ingroup covariance
 * @{
 */
package com.intel.daal.algorithms.covariance;

import java.lang.annotation.Native;

import com.intel.daal.utils.*;
/**
 * <a name="DAAL-CLASS-ALGORITHMS__COVARIANCE__METHOD"></a>
 * @brief Available methods for computing the correlation or variance-covariance matrix
 */
public final class Method {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    private int _value;

    /**
     * Constructs the method object using the provided value
     * @param value     Value corresponding to the method object
     */
    public Method(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the method object
     * @return Value corresponding to the method object
     */
    public int getValue() {
        return _value;
    }

    @Native private static final int DefaultDense    = 0;
    @Native private static final int SinglePassDense = 1;
    @Native private static final int SumDense        = 2;
    @Native private static final int FastCSR         = 3;
    @Native private static final int SinglePassCSR   = 4;
    @Native private static final int SumCSR          = 5;

    public static final Method defaultDense    = new Method(DefaultDense);    /*!< Default: performance-oriented method.
                                                                                   Works with all types of numeric tables */
    public static final Method singlePassDense = new Method(SinglePassDense); /*!< Single-pass: implementation of
                                                                                   the single-pass algorithm proposed by D.H.D. West.
                                                                                   Works with all types of numeric tables */
    public static final Method sumDense        = new Method(SumDense);        /*!< Precomputed sum: implementation of the moments computation
                                                                                   algorithm in the case of a precomputed sum.
                                                                                   Works with all types of numeric tables */
    public static final Method fastCSR         = new Method(FastCSR);         /*!< Default: performance-oriented method.
                                                                                   Works with compressed sparse row (CSR)
                                                                                   numeric tables */
    public static final Method singlePassCSR   = new Method(SinglePassCSR);   /*!< Single-pass: implementation of
                                                                                   the single-pass algorithm proposed by D.H.D. West.
                                                                                   Works with CSR numeric tables */
    public static final Method sumCSR          = new Method(SumCSR);          /*!< Precomputed sum: implementation of the
                                                                                   algorithm in the case of a precomputed sum.
                                                                                   Works with CSR numeric tables */
}
/** @} */
