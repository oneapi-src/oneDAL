/* file: TrainParameter.java */
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
