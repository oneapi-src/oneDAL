/* file: PredictionResultId.java */
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

package com.intel.daal.algorithms.classifier.prediction;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__CLASSIFIER__PREDICTION__PREDICTIONRESULTID"></a>
 * @brief Available identifiers of results of the classifier model-based prediction algorithm
 */
public final class PredictionResultId {
    private int _value;

    public PredictionResultId(int value) {
        _value = value;
    }

    public int getValue() {
        return _value;
    }

    private static final int Prediction = 0;

    /** Prediction results */
    public static final PredictionResultId prediction = new PredictionResultId(Prediction);

}
