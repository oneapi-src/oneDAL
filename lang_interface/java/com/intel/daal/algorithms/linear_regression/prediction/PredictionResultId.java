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

package com.intel.daal.algorithms.linear_regression.prediction;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LINEAR_REGRESSION__PREDICTION__PREDICTIONRESULTID"></a>
 * @brief Available identifiers of the result of linear regression model-based prediction
 */
public final class PredictionResultId {
    private int _value;

    static {
        System.loadLibrary("JavaAPI");
    }
/** Default constructor */
    public PredictionResultId(int value) {
        _value = value;
    }

    /**
     * Returns a value corresponding to the identifier of the input object
     * \return Value corresponding to the identifier
     */

    public int getValue() {
        return _value;
    }

    private static final int PredictionId = 0;

    /** Result of linear regression model-based prediction */
    public static final PredictionResultId prediction = new PredictionResultId(PredictionId);
}
