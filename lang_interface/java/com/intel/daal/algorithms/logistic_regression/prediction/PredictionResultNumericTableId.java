/* file: PredictionResultNumericTableId.java */
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
 * @ingroup logistic_regression_prediction
 * @{
 */
package com.intel.daal.algorithms.logistic_regression.prediction;

import java.lang.annotation.Native;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.classifier.prediction.PredictionResultId;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LOGISTIC_REGRESSION__PREDICTION__PREDICTIONRESULTNUMERICTABLEID"></a>
 * @brief Available identifiers of the result of logistic regression model-based prediction
 */
public final class PredictionResultNumericTableId {
    private int _value;

    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the prediction result object identifier using the provided value
     * @param value     Value corresponding to the prediction result object identifier
     */
    public PredictionResultNumericTableId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the prediction result object identifier
     * @return Value corresponding to the prediction result object identifier
     */
    public int getValue() {
        return _value;
    }

    @Native private static final int probabilitiesValue = 1;            /*!< Numeric table of size: n x 1, if nClasses = 2, n x nClasses, if nClasses > 2
                                                                     containing probabilities of classes computed when
                                                                     computeClassesProbabilities option is enabled.
                                                                     In case  nClasses = 2 the table contains probabilities of class _1. */
    @Native private static final int logProbabilitiesValue = 2;         /*!< Numeric table of size: n x 1, if nClasses = 2, n x nClasses, if nClasses > 2
                                                                     containing logarithms of classes_ probabilities computed when
                                                                     computeClassesLogProbabilities option is enabled.
                                                                     In case nClasses = 2 the table contains logarithms of class _1_ probabilities. */

    /** Result of logistic regression model-based prediction */
    public static final PredictionResultNumericTableId probabilities = new PredictionResultNumericTableId(probabilitiesValue);
    public static final PredictionResultNumericTableId logProbabilities = new PredictionResultNumericTableId(logProbabilitiesValue);
}
/** @} */
