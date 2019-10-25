/* file: LossFunctionType.java */
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
 * @ingroup gbt_distributed
 * @{
 */
package com.intel.daal.algorithms.gbt.regression.training;

import com.intel.daal.utils.*;
/**
 * <a name="DAAL-ENUM-ALGORITHMS__GBT__REGRESSION__TRAINING__LOSS_FUNCTION_TYPE"></a>
 * @brief Loss function type
 */
public final class LossFunctionType {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    private int _value;

    /**
     * Constructs the Loss function type object using the provided value
     * @param value     Value corresponding to the loss function type object
     */
    public LossFunctionType(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the loss function type object
     * @return Value corresponding to the loss function type object
     */
    public int getValue() {
        return _value;
    }

    private static final int squaredValue = 0;
    private static final int customValue = 1;

    public static final LossFunctionType squared = new LossFunctionType(squaredValue); /*!< L(y,f) = ([y-f(x)]^2)/2 */
    public static final LossFunctionType custom = new LossFunctionType(customValue); /*!< Should be differentiable up to the second order */
}
/** @} */
