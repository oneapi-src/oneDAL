/* file: ResultCollection.java */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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
 * @ingroup pca_quality_metric_set
 * @{
 */
package com.intel.daal.algorithms.pca.quality_metric_set;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.services.DaalContext;
import com.intel.daal.algorithms.pca.quality_metric.ExplainedVarianceResult;
import com.intel.daal.algorithms.quality_metric.QualityMetricResult;

/**
 *  <a name="DAAL-CLASS-ALGORITHMS__PCA__QUALITY_METRIC_SET__RESULTCOLLECTION"></a>
 *  @brief Class that implements functionality of the collection of result objects of the quality metrics algorithm
 */
public class ResultCollection extends com.intel.daal.algorithms.quality_metric_set.ResultCollection {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public ResultCollection(DaalContext context, long cAlgorithm, ComputeMode cmode) {
        super(context, cAlgorithm, cmode);
    }

    /**
     * Returns the element that matches the identifier
     * @param  id     Identifier of the quality metric
     * @return Result object
     */
    public QualityMetricResult getResult(QualityMetricId id) {
        if (id == QualityMetricId.explainedVariancesMetrics) {
            return new ExplainedVarianceResult(getContext(), cGetResult(getCObject(), id.getValue()));
        }
        throw new IllegalArgumentException("id unsupported");
    }
}
/** @} */
