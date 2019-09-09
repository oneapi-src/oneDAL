/* file: ExplainedVarianceResultId.java */
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
 * @ingroup pca_quality_metric_explained_variance
 * @{
 */
package com.intel.daal.algorithms.pca.quality_metric;

import java.lang.annotation.Native;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__QUALITY_METRIC__EXPLAINEDVARIANCERESULTTID"></a>
 * @brief Available identifiers of the result of explained variance quality metrics
 */
public final class ExplainedVarianceResultId {
    private int _value;

    /**
     * Constructs the result object identifier using the provided value
     * @param value     Value corresponding to the result object identifier
     */
    public ExplainedVarianceResultId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the result object identifier
     * @return Value corresponding to the result object identifier
     */
    public int getValue() {
        return _value;
    }

    @Native private static final int explainedVariancesId = 0;
    @Native private static final int explainedVariancesRatiosId = 1;
    @Native private static final int noiseVarianceId = 2;

    /*!< Explained variances */
    public static final ExplainedVarianceResultId explainedVariances       = new ExplainedVarianceResultId(explainedVariancesId);
    /*!< Explained variances ratios */
    public static final ExplainedVarianceResultId explainedVariancesRatios = new ExplainedVarianceResultId(explainedVariancesRatiosId);
    /*!< Noise variance */
    public static final ExplainedVarianceResultId noiseVariance            = new ExplainedVarianceResultId(noiseVarianceId);
}
/** @} */
