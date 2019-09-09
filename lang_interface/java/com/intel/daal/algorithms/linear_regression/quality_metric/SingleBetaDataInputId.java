/* file: SingleBetaDataInputId.java */
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
 * @defgroup linear_regression_quality_metric_single_beta Single Beta Coefficient
 * @ingroup linear_regression_quality_metric_set
 * @{
 */
package com.intel.daal.algorithms.linear_regression.quality_metric;

import java.lang.annotation.Native;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LINEAR_REGRESSION__QUALITY_METRIC__SINGLEBETADATAINPUTID"></a>
 * @brief Available identifiers of input objects for a single beta quality metrics
 */
public final class SingleBetaDataInputId {
    private int _value;

    /**
     * Constructs the input object identifier using the provided value
     * @param value     Value corresponding to the input object identifier
     */
    public SingleBetaDataInputId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the input object identifier
     * @return Value corresponding to the input object identifier
     */
    public int getValue() {
        return _value;
    }

    @Native private static final int ExpectedResponses   = 0;
    @Native private static final int PredictedResponses = 1;

    /*!< Expected responses (Y), dependent variables */
    public static final SingleBetaDataInputId expectedResponses   = new SingleBetaDataInputId(ExpectedResponses);
    /*!< Predicted responses (Z) */
    public static final SingleBetaDataInputId predictedResponses = new SingleBetaDataInputId(PredictedResponses);
}
/** @} */
