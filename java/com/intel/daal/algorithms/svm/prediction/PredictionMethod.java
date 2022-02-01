/* file: PredictionMethod.java */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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
 * @ingroup svm_prediction
 * @{
 */
package com.intel.daal.algorithms.svm.prediction;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__SVM__PREDICTION__PREDICTIONMETHOD"></a>
 * @brief Available methods to compute the results of the SVM prediction algorithm
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
     * Returns the value corresponding to prediction the method object
     * @return Value corresponding to the prediction method object
     */
    public int getValue() {
        return _value;
    }

    private static final int DefaultDense = 0;

    public static final PredictionMethod defaultDense = new PredictionMethod(
            DefaultDense); /*!< Default SVM prediction method */
}
/** @} */
