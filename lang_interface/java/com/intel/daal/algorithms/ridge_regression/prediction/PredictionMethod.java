/* file: PredictionMethod.java */
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

package com.intel.daal.algorithms.ridge_regression.prediction;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__RIDGE_REGRESSION__PREDICTION__PREDICTIONMETHOD"></a>
 * @brief Available methods of ridge regression model-based prediction
 */
public final class PredictionMethod {

    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    private int _value;

    /** Default constructor */
    public PredictionMethod(int value) {
        _value = value;
    }

    /**
     * Returns a value corresponding to the identifier of the input object
     * \return Value corresponding to the identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int defaultDenseValue = 0;

    public static final PredictionMethod defaultDense = new PredictionMethod(
            defaultDenseValue); /*!< Default method of ridge regression model-based prediction */
}
