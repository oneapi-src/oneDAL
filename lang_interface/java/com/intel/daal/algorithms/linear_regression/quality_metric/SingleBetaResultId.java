/* file: SingleBetaResultId.java */
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
 * <a name="DAAL-CLASS-ALGORITHMS__LINEAR_REGRESSION__QUALITY_METRIC__SINGLEBETARESULTID"></a>
 * @brief Available identifiers of the result of single beta quality metrics
 */
public final class SingleBetaResultId {
    private int _value;

    /**
     * Constructs the result object identifier using the provided value
     * @param value     Value corresponding to the result object identifier
     */
    public SingleBetaResultId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the result object identifier
     * @return Value corresponding to the result object identifier
     */
    public int getValue() {
        return _value;
    }

    @Native private static final int Rms   = 0;
    @Native private static final int Variance = 1;
    @Native private static final int ZScore = 2;
    @Native private static final int ConfidenceIntervals = 3;
    @Native private static final int InverseOfXtX = 4;

    /*!< Root means square errors computed for each response (dependent variable) */
    public static final SingleBetaResultId rms   = new SingleBetaResultId(Rms);
    /*!< Variance computed for each response (dependent variable) */
    public static final SingleBetaResultId variance = new SingleBetaResultId(Variance);
    /*!< Z-score statistics used in testing of insignificance one beta coefficient. H0: beta[i]=0 */
    public static final SingleBetaResultId zScore = new SingleBetaResultId(ZScore);
    /*!< Limits of the confidence intervals for each beta */
    public static final SingleBetaResultId confidenceIntervals = new SingleBetaResultId(ConfidenceIntervals);
    /*!< Inverse(Xt * X) matrix */
    public static final SingleBetaResultId inverseOfXtX = new SingleBetaResultId(InverseOfXtX);
}
/** @} */
