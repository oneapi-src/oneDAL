/* file: InputDataCollection.java */
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
 * @defgroup pca_quality_metric_set Quality Metrics
 * @brief Contains classes to check the quality of the model trained with the PCA algorithm
 * @ingroup pca
 * @{
 */
package com.intel.daal.algorithms.pca.quality_metric_set;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Input;
import com.intel.daal.services.DaalContext;
import com.intel.daal.algorithms.pca.quality_metric.ExplainedVarianceInput;
import com.intel.daal.algorithms.quality_metric.QualityMetricInput;

/**
 *  <a name="DAAL-CLASS-ALGORITHMS__PCA__QUALITY_METRIC_SET__INPUTDATACOLLECTION"></a>
 *  @brief Class that implements functionality of the collection of input objects for the quality metrics algorithm
 */
public class InputDataCollection extends com.intel.daal.algorithms.quality_metric_set.InputDataCollection {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public InputDataCollection(DaalContext context, long cAlgorithm, ComputeMode cmode) {
        super(context, cAlgorithm, cmode);
    }

    /**
     * Returns the element that matches the identifier
     * @param  id    Identifier of the quality metric
     * @return Input object
     */
    public QualityMetricInput getInput(QualityMetricId id) {
        if (id == QualityMetricId.explainedVariancesMetrics) {
            return new ExplainedVarianceInput(getContext(), cGetInput(getCObject(), id.getValue()));
        }
        throw new IllegalArgumentException("id unsupported");
    }
}
/** @} */
