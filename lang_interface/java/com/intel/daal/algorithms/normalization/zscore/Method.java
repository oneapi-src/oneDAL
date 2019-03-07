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
 * @ingroup zscore
 * @{
 */
package com.intel.daal.algorithms.normalization.zscore;

import com.intel.daal.utils.*;
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NORMALIZATION__ZSCORE__METHOD"></a>
 * @brief Available methods for Z-score normalization
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

    private static final int   DefaultDense = 0;
    private static final int   SumDense = 1;
    public static final Method defaultDense       = new Method(DefaultDense); /*!< Default: performance-oriented method.
                                                                              Works with all types  of input numeric tables */
    public static final Method sumDense           = new Method(SumDense);    /*!< Precomputed sum: implementation of computation
                                                                              algorithm in the case of a precomputed sum.
                                                                              Works with all types of input numeric tables */
}
/** @} */
