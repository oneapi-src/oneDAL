/* file: InputDataCollection.java */
/*******************************************************************************
* Copyright 2014-2022 Intel Corporation
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
 * @defgroup linear_regression_quality_metric_set Quality Metrics
 * @brief Contains classes to check the quality of the model trained with the linear regression algorithm
 * @ingroup linear_regression
 * @{
 */
package com.intel.daal.algorithms.linear_regression.quality_metric_set;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.linear_regression.quality_metric.SingleBetaInput;
import com.intel.daal.algorithms.linear_regression.quality_metric.GroupOfBetasInput;
import com.intel.daal.services.DaalContext;
import com.intel.daal.algorithms.quality_metric.QualityMetricInput;
/**
 *  <a name="DAAL-CLASS-ALGORITHMS__LINEAR_REGRESSION__QUALITY_METRIC_SET__INPUTDATACOLLECTION"></a>
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
        if (id == QualityMetricId.singleBeta)
            return new SingleBetaInput(getContext(), cGetInput(getCObject(), id.getValue()));
        if (id == QualityMetricId.groupOfBetas)
            return new GroupOfBetasInput(getContext(), cGetInput(getCObject(), id.getValue()));
        throw new IllegalArgumentException("id unsupported");
    }
}
/** @} */
