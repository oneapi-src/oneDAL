/* file: QualityMetricSetBatch.java */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
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
 * @defgroup adaboost_quality_metric_set_batch Batch
 * @ingroup adaboost_quality_metric_set
 * @{
 */
/**
 * @brief Contains classes to check the quality of the model trained with the AdaBoost algorithm
 */
package com.intel.daal.algorithms.adaboost.quality_metric_set;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__ADABOOST__QUALITY_METRIC_SET__QUALITYMETRICSETBATCH"></a>
 * @brief Class that represents a quality metric set to check the model trained with the AdaBoost algorithm
 *
 * @par Enumerations
 *      - @ref QualityMetricId  Identifiers of quality metrics provided by the library
 */
public class QualityMetricSetBatch extends com.intel.daal.algorithms.quality_metric_set.QualityMetricSetBatch {
    private InputDataCollection inputData;

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the quality metric set
     * @param context   Context to manage the quality metric set
     */
    public QualityMetricSetBatch(DaalContext context) {
        super(context);
        this.cObject = cInit();
        inputData = new InputDataCollection(getContext(), cObject, ComputeMode.batch);
    }

    /**
     * Returns the collection of input objects of quality metrics algorithms
     * @return Collection of input objects of quality metrics algorithms
     */
    public InputDataCollection getInputDataCollection() {
        return inputData;
    }

    /**
     * Computes the results for the quality metric set in the batch processing mode
     * @return Structure that contains a computed quality metric set
     */
    @Override
    public ResultCollection compute() {
        super.compute();
        return new ResultCollection(getContext(), cObject, ComputeMode.batch);
    }

    private native long cInit();
}
/** @} */
