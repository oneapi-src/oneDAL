/* file: PredictionMethod.java */
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
 * @ingroup multi_class_classifier_prediction
 * @{
 */
package com.intel.daal.algorithms.multi_class_classifier.prediction;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTI_CLASS_CLASSIFIER__PREDICTION__PREDICTIONMETHOD"></a>
 * @brief Available methods for the multi-class classifier prediction
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

    private static final int multiClassClassifierWuValue = 0;
    private static final int voteBasedValue = 1;

    public static final PredictionMethod multiClassClassifierWu = new PredictionMethod(
            multiClassClassifierWuValue); /*!< Prediction method for the Multi-class classifier proposed by Ting-Fan Wu et al */

    public static final PredictionMethod voteBased = new PredictionMethod(
            voteBasedValue); /*!< Prediction method that is based on votes returned by two-class classifiers */
}
/** @} */
