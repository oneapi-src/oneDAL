/* file: PredictionInputId.java */
/*******************************************************************************
* Copyright 2014-2022 Intel Corporation
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
 * @ingroup lasso_regression_prediction
 * @{
 */
package com.intel.daal.algorithms.lasso_regression.prediction;

import java.lang.annotation.Native;

import com.intel.daal.utils.*;
/**
 * <a name="DAAL-CLASS-ALGORITHMS__LASSO_REGRESSION__PREDICTION__PREDICTIONINPUTID"></a>
 * @brief Available identifiers of input objects for lasso regression model-based prediction
 */
public final class PredictionInputId {
    private int _value;

    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the input object identifier using the provided value
     * @param value     Value corresponding to the input object identifier
     */
    public PredictionInputId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the input object identifier
     * @return Value corresponding to the input object identifier
     */
    public int getValue() {
        return _value;
    }

    @Native private static final int dataId  = 0;
    @Native private static final int modelId = 1;

    /** %Input data table */
    public static final PredictionInputId data  = new PredictionInputId(dataId);
    /** Trained lasso regression model */
    public static final PredictionInputId model = new PredictionInputId(modelId);
}
/** @} */
