/* file: Parameter.java */
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
 * @ingroup gbt_regression_prediction
 */
/**
 * @brief Contains parameter for gradient boosted trees regression prediction algorithm
 */
package com.intel.daal.algorithms.gbt.regression.prediction;

import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT__REGRESSION__PREDICTION__PARAMETER"></a>
 * @brief Parameter of the gradient boosted trees regression prediction algorithm
 */
public class Parameter extends com.intel.daal.algorithms.Parameter {

    public Parameter(DaalContext context, long cParameter) {
        super(context, cParameter);
    }

    /**
     * Returns number of iterations of the trained model to be used for prediction
     * by the gradient boosted trees prediction algorithm
     * @return Number of iterations
     */
    public long getNIterations() {
        return cGetNIterations(this.cObject);
    }

    /**
     * Sets the number of iterations of the trained model to be used for prediction
     * by the gradient boosted trees prediction algorithm
     * @param n Number of iterations
     */
    public void setNIterations(long n) {
        cSetNIterations(this.cObject, n);
    }

    private native long cGetNIterations(long parAddr);
    private native void cSetNIterations(long parAddr, long value);
}
/** @} */
