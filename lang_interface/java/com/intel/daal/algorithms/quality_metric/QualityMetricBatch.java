/* file: QualityMetricBatch.java */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

/**
 * @defgroup quality_metric Quality Metrics
 * @brief Contains classes for checking the quality of the classification algorithms
 * @ingroup analysis
 * @{
 */
/**
 * @defgroup quality_metric_batch Batch
 * @ingroup quality_metric
 * @{
 */
/**
 * @brief Contains classes to compute quality metrics
 */
package com.intel.daal.algorithms.quality_metric;

import com.intel.daal.utils.*;
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
        LibUtils.loadLibrary();
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
/** @} */
/** @} */
