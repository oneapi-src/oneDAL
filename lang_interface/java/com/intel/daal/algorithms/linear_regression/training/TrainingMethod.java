/* file: TrainingMethod.java */
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
 * @ingroup linear_regression_training
 * @{
 */
package com.intel.daal.algorithms.linear_regression.training;

import java.lang.annotation.Native;

import com.intel.daal.utils.*;
/**
 * <a name="DAAL-CLASS-ALGORITHMS__LINEAR_REGRESSION__TRAINING__TRAININGMETHOD"></a>
 * @brief Available methods for linear regression model-based training
 */
public final class TrainingMethod {

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    private int _value;

    /**
     * Constructs the training method object using the provided value
     * @param value     Value corresponding to the training method object
     */
    public TrainingMethod(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the training method object
     * @return Value corresponding to the training method object
     */
    public int getValue() {
        return _value;
    }

    @Native private static final int defaultDenseValue = 0;
    @Native private static final int normEqDenseValue  = 0;
    @Native private static final int qrDenseValue      = 1;

    public static final TrainingMethod defaultDense = new TrainingMethod(
            defaultDenseValue);                                          /*!< Normal equations method */
    public static final TrainingMethod normEqDense  = new TrainingMethod(
            normEqDenseValue);                                           /*!< Normal equations method */
    public static final TrainingMethod qrDense      = new TrainingMethod(
            qrDenseValue);                                               /*!< QR decomposition-based method */
}
/** @} */
