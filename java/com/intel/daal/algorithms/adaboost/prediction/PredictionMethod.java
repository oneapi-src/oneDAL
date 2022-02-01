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
 * @ingroup adaboost_prediction
 * @{
 */
package com.intel.daal.algorithms.adaboost.prediction;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__ADABOOST__PREDICTION__PREDICTIONMETHOD"></a>
 * @brief Available methods for predictions based on the AdaBoost model
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

    private static final int Samme = 0;
    private static final int DefaultDense = Samme;
    private static final int SammeR = 1;

    /** Default AdaBoost training method */
    public static final PredictionMethod defaultDense = new PredictionMethod(DefaultDense);
    /** SAMME algorithm */
    public static final PredictionMethod samme = new PredictionMethod(Samme);
    /** SAMME.R algorithm, need probabilities from weak learner prediction result */
    public static final PredictionMethod sammeR = new PredictionMethod(SammeR);
}
/** @} */
