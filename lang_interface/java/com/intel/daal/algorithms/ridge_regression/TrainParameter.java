/* file: TrainParameter.java */
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
 * @ingroup ridge_regression
 * @{
 */
/**
 * \brief Contains classes for computing the result of the ridge regression algorithm
 */
package com.intel.daal.algorithms.ridge_regression;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

import com.intel.daal.algorithms.ridge_regression.Parameter;

import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.Factory;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__RIDGE_REGRESSION__TRAINPARAMETER"></a>
 * @brief Ridge regression algorithm parameters
 */
public class TrainParameter extends com.intel.daal.algorithms.ridge_regression.Parameter {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public TrainParameter(DaalContext context, long cParameter) {
        super(context, cParameter);
    }

    /**
     * Sets the numeric table that represents ridge parameters. If no ridge parameters are provided,
     * the implementation will generate ridge parameters equal to 1.
     * @param ridgeParameters The numeric table that represents ridge parameters
     */
    public void setRidgeParameters(NumericTable ridgeParameters) {
        cSetRidgeParameters(this.cObject, ridgeParameters.getCObject());
    }

    /**
     * Retrieves the numeric table that represents ridge parameters. If no ridge parameters are provided,
     * the implementation will generate ridge parameters equal to 1.
     * @return The numeric table that represents ridge parameters.
     */
    public NumericTable getRidgeParameters() {
        return (NumericTable)Factory.instance().createObject(getContext(), cGetRidgeParameters(this.cObject));
    }

    private native void cSetRidgeParameters(long parAddr, long ridgeParameters);
    private native long cGetRidgeParameters(long parAddr);
}
/** @} */
