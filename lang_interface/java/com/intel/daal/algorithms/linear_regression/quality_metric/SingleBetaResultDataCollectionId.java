/* file: SingleBetaResultDataCollectionId.java */
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
 * @ingroup linear_regression_quality_metric_single_beta
 * @{
 */
package com.intel.daal.algorithms.linear_regression.quality_metric;

import java.lang.annotation.Native;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LINEAR_REGRESSION__QUALITY_METRIC__SINGLEBETARESULTDATACOLLECTIONID"></a>
 * @brief Available identifiers of the result of single beta quality metrics
 */
public final class SingleBetaResultDataCollectionId {
    private int _value;

    /**
     * Constructs the result object identifier using the provided value
     * @param value     Value corresponding to the result object identifier
     */
    public SingleBetaResultDataCollectionId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the result object identifier
     * @return Value corresponding to the result object identifier
     */
    public int getValue() {
        return _value;
    }

    @Native private static final int BetaCovariances   = 5;

    /*!< Numeric tables with nBeta x nBeta variance-covariance matrix for betas of each response (dependent variable) */
    public static final SingleBetaResultDataCollectionId betaCovariances   = new SingleBetaResultDataCollectionId(BetaCovariances);
}
/** @} */
