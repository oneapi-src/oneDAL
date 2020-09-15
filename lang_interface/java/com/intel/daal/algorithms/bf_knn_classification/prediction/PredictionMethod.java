/* file: PredictionMethod.java */
/*******************************************************************************
* Copyright 2014-2020 Intel Corporation
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
 * @ingroup bf_knn_classification_prediction
 * @{
 */
package com.intel.daal.algorithms.bf_knn_classification.prediction;

import java.lang.annotation.Native;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__BF_KNN_CLASSIFICATION__PREDICTION__PREDICTIONMETHOD"></a>
 * @brief Available methods for the brute-force k nearest neighbors prediction
 */
public final class PredictionMethod {

    private int _value;

    /**
     * Constructs the prediction method object using the provided value
     * @param value     Value corresponding to the prediction method object
     */
    public PredictionMethod(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the prediction method object
     * @return Value corresponding to the prediction method object
     */
    public int getValue() {
        return _value;
    }

    @Native private static final int defaultDenseValue = 0;

    public static final PredictionMethod defaultDense = new PredictionMethod(defaultDenseValue); /*!< Default method */
}
/** @} */
