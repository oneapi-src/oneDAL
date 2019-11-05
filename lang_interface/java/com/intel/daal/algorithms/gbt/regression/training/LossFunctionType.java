/* file: LossFunctionType.java */
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
