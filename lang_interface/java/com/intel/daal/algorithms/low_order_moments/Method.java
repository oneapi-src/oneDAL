/* file: Method.java */
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

/**
 * @ingroup low_order_moments
 * @{
 */
package com.intel.daal.algorithms.low_order_moments;

import com.intel.daal.utils.*;
/**
 * <a name="DAAL-CLASS-ALGORITHMS__LOW_ORDER_MOMENTS__METHOD"></a>
 * @brief Available methods for computing moments of low order %Moments
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
/** @} */
