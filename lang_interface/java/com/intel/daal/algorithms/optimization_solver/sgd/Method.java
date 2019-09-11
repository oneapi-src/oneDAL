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
 * @ingroup sgd
 * @{
 */
package com.intel.daal.algorithms.optimization_solver.sgd;

import com.intel.daal.utils.*;
/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__SGD__METHOD"></a>
 * @brief Available methods for computing the SGD algorithm
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

    private static final int defaultDenseId = 0;
    private static final int miniBatchId = 1;
    private static final int momentumId = 2;

    public static final Method defaultDense = new Method(defaultDenseId); /*!< Default method */
    public static final Method miniBatch = new Method(miniBatchId); /*!< Mini-batch method */
    public static final Method momentum = new Method(momentumId); /*!< Momentum method */
}
/** @} */
