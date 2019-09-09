/* file: ModelInputId.java */
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
 * @defgroup decision_forest_regression_prediction Prediction
 * @brief Contains classes for making prediction based on the decision_forest regression model
 * @ingroup decision_forest_regression
 * @{
 */
/**
 * @brief Contains classes for making prediction based on the decision_forest regression model
 */
package com.intel.daal.algorithms.decision_forest.regression.prediction;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DECISION_FOREST__REGRESSION__PREDICTION__MODELINPUTID"></a>
 * @brief Available identifiers of input objects of the decision_forest regression predication algorithm
 */
public final class ModelInputId {
    private int _value;

    /**
     * Constructs the model input object identifier using the provided value
     * @param value     Value corresponding to the model input object identifier
     */
    public ModelInputId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the model input object identifier
     * @return Value corresponding to the model input object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int Model = 1;

    public static final ModelInputId model = new ModelInputId(Model); /*!< Model to use in the prediction stage*/
}
/** @} */
