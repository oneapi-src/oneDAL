/* file: GroupOfBetasInput.java */
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
 * @defgroup linear_regression_quality_metric_group_of_betas Group of Beta Coefficients
 * @ingroup linear_regression_quality_metric_set
 * @{
 */
package com.intel.daal.algorithms.linear_regression.quality_metric;

import com.intel.daal.utils.*;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LINEAR_REGRESSION__QUALITY_METRIC__GROUPOFBETASINPUT"></a>
 * @brief  Class for the input objects of the algorithm
 */
public class GroupOfBetasInput extends com.intel.daal.algorithms.quality_metric.QualityMetricInput {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public GroupOfBetasInput(DaalContext context, long cObject) {
        super(context);
        this.cObject = cObject;
    }

    /**
     * Sets the input object for linear regression quality metric
     * @param id    Identifier of the input object
     * @param val   Value of the input object
     */
    public void set(GroupOfBetasInputId id, NumericTable val) {
        if (id != GroupOfBetasInputId.expectedResponses
                && id != GroupOfBetasInputId.predictedResponses
                && id != GroupOfBetasInputId.predictedReducedModelResponses) {
            throw new IllegalArgumentException("id unsupported");
        }

        cSetInputTable(cObject, id.getValue(), val.getCObject());
    }

    /**
     * Returns the input object for linear regression quality metric
     * @param id Identifier of the input object
     * @return   Input object that corresponds to the given identifier
     */
    public NumericTable get(GroupOfBetasInputId id) {
        if (id == GroupOfBetasInputId.expectedResponses
            || id == GroupOfBetasInputId.predictedResponses
            || id == GroupOfBetasInputId.predictedReducedModelResponses) {
            return (NumericTable)Factory.instance().createObject(getContext(), cGetInputTable(cObject, id.getValue()));
        } else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    private native void cSetInputTable(long inputAddr, int id, long ntAddr);
    private native long cGetInputTable(long cInput, int id);
}
/** @} */
