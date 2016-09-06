/* file: Method.java */
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

package com.intel.daal.algorithms.low_order_moments;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LOW_ORDER_MOMENTS__METHOD"></a>
 * @brief Available methods for computing moments of low order %Moments
 */
public final class Method {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    private int _value;

    public Method(int value) {
        _value = value;
    }

    public int getValue() {
        return _value;
    }

    private static final int DefaultDense    = 0;
    private static final int SinglePassDense = 1;
    private static final int SumDense        = 2;
    private static final int FastCSR         = 3;
    private static final int SinglePassCSR   = 4;
    private static final int SumCSR          = 5;

    public static final Method defaultDense    = new Method(DefaultDense);    /*!< Default: performance-oriented method.
                                                                              Works with all types
                                                                              of input numeric tables */
    public static final Method singlePassDense = new Method(SinglePassDense); /*!< Single-pass: implementation of
                                                                              the single-pass algorithm proposed by D.H.D. West.
                                                                              Works with all types of
                                                                              input numeric tables */
    public static final Method sumDense        = new Method(SumDense);        /*!< Precomputed sum: implementation of moments computation
                                                                              algorithm in the case of a precomputed sum.
                                                                              Works with all types of input numeric tables */
    public static final Method fastCSR         = new Method(FastCSR);         /*!< Default: performance-oriented method.
                                                                                   Works with Compressed Sparse Rows(CSR)
                                                                                   input numeric tables */
    public static final Method singlePassCSR   = new Method(SinglePassCSR);   /*!< Single-pass: implementation of
                                                                                   the single-pass algorithm proposed by D.H.D. West.
                                                                                   Works with Compressed Sparse Rows(CSR)
                                                                                   input numeric tables */
    public static final Method sumCSR          = new Method(SumCSR);          /*!< Precomputed sum: implementation of moments
                                                                              computation algorithm in the case of a precomputed sum.
                                                                              Works with Compressed Sparse Rows(CSR)
                                                                              input numeric tables */
}
