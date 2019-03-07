/* file: ResultCollection.java */
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
 * @ingroup svm_quality_metric_set
 * @{
 */
package com.intel.daal.algorithms.svm.quality_metric_set;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.classifier.quality_metric.binary_confusion_matrix.BinaryConfusionMatrixResult;
import com.intel.daal.services.DaalContext;

/**
 *  <a name="DAAL-CLASS-ALGORITHMS__SVM__QUALITY_METRIC_SET__RESULTCOLLECTION"></a>
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
    public BinaryConfusionMatrixResult getResult(QualityMetricId id) {
        if (id != QualityMetricId.confusionMatrix) {
            throw new IllegalArgumentException("id unsupported");
        }
        return new BinaryConfusionMatrixResult(getContext(), cGetResult(getCObject(), id.getValue()));
    }
}
/** @} */
