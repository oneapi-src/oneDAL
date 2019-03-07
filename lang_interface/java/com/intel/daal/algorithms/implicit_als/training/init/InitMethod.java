/* file: InitMethod.java */
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
 * @ingroup implicit_als_init
 * @{
 */
package com.intel.daal.algorithms.implicit_als.training.init;

import com.intel.daal.utils.*;
/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__INIT__INITMETHOD"></a>
 * @brief Available methods for computing initial values for the implicit ALS training algorithm
 */
public final class InitMethod {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    private int _value;

    /**
     * Constructs the initialization method object using the provided value
     * @param value     Value corresponding to the initialization method object
     */
    public InitMethod(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the initialization method object
     * @return Value corresponding to the initialization method object
     */
    public int getValue() {
        return _value;
    }

    private static final int defaultDenseId = 0;
    private static final int fastCSRId      = 1;

    /** Method for initializing the implicit ALS algorithm */
    public static final InitMethod defaultDense = new InitMethod(
            defaultDenseId);                                     /*!< Default: initialization method for input data stored
                                                                    in the dense format */
    /** Method for initializing the implicit ALS algorithm */
    public static final InitMethod fastCSR      = new InitMethod(
            fastCSRId);                                          /*!< Initialization method for input data stored
                                                                    in the compressed sparse row (CSR) format */
}
/** @} */
