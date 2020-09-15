/* file: PredictionResultId.java */
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

import com.intel.daal.utils.*;
/**
 * <a name="DAAL-CLASS-ALGORITHMS__BF_KNN_CLASSIFICATION__PREDICTION__PREDICTIONRESULTID"></a>
 * @brief Available identifiers of the result of brute-force knn model-based prediction
 */
public final class PredictionResultId {
    private int _value;

    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the prediction result object identifier using the provided value
     * @param value     Value corresponding to the prediction result object identifier
     */
    public PredictionResultId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the prediction result object identifier
     * @return Value corresponding to the prediction result object identifier
     */
    public int getValue() {
        return _value;
    }

    @Native private static final int PredictionId = 0;
    @Native private static final int IndicesId    = 3;
    @Native private static final int DistancesId  = 4;

    /** Prediction results */
    public static final PredictionResultId prediction = new PredictionResultId(PredictionId);
    /** Indices of nearest neighbors */
    public static final PredictionResultId indices    = new PredictionResultId(IndicesId);
    /** Distances to nearest neighbors */
    public static final PredictionResultId distances  = new PredictionResultId(DistancesId);
}
/** @} */
