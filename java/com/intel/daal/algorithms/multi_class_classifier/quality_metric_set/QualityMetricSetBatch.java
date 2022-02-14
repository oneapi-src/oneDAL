/* file: QualityMetricSetBatch.java */
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
 * @defgroup multi_class_classifier_quality_metric_set_batch Batch
 * @ingroup multi_class_classifier_quality_metric_set
 * @{
 */
/**
 * @brief Contains classes to check the quality of the model trained with the multi-class SVM algorithm
 */
package com.intel.daal.algorithms.multi_class_classifier.quality_metric_set;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTI_CLASS_CLASSIFIER__QUALITY_METRIC_SET__QUALITYMETRICSETBATCH"></a>
 * @brief Class that represents a quality metric set to check the model trained with the multi-class SVM algorithm
 *
 * @par Enumerations
 *      - @ref QualityMetricId  Identifiers of quality metrics provided by the library
 */

public class QualityMetricSetBatch extends com.intel.daal.algorithms.quality_metric_set.QualityMetricSetBatch {
    public QualityMetricSetParameter parameter;
    private InputDataCollection      inputData;

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public QualityMetricSetBatch(DaalContext context, long nClasses) {
        super(context);
        this.cObject = cInit(nClasses);
        inputData = new InputDataCollection(getContext(), cObject, ComputeMode.batch);
        parameter = new QualityMetricSetParameter(getContext(), cInitParameter(cObject), nClasses);
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

    private native long cInit(long nClasses);

    private native long cInitParameter(long algAddr);
}
/** @} */
