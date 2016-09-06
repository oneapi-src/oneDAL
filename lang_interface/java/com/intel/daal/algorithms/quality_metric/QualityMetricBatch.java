/* file: QualityMetricBatch.java */
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

/**
 * @brief Contains classes to compute quality metrics
 */
package com.intel.daal.algorithms.quality_metric;

import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__QUALITY_METRIC__QUALITYMETRICBATCH"></a>
 * @brief Provides methods to compute quality metrics of an algorithm in the batch processing mode.
 *        Quality metric is a numerical characteristic or a set of connected numerical characteristics
 *        that represents the qualitative aspect of a computed statistical estimate, model,
 *        or decision-making result.
 */
public abstract class QualityMetricBatch extends com.intel.daal.algorithms.AnalysisBatch {
    protected Precision prec;

    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs quality metric algorithm
     * @param context   Context to manage the quality metric algorithm
     */
    public QualityMetricBatch(DaalContext context) {
        super(context);
    }

    /**
     * Returns the newly allocated quality metric algorithm with a copy of input objects
     * and parameters of this algorithm
     * @param context   Context to manage the quality metric algorithm
     *
     * @return The newly allocated algorithm
     */
    @Override
    public abstract QualityMetricBatch clone(DaalContext context);
}
