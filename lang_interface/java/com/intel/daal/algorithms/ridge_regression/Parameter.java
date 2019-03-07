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
 * @ingroup ridge_regression
 * @{
 */
/**
 * \brief Contains classes for computing the result of the ridge regression algorithm
 */
package com.intel.daal.algorithms.ridge_regression;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__RIDGE_REGRESSION__PARAMETER"></a>
 * @brief Ridge regression algorithm parameters
 */
public class Parameter extends com.intel.daal.algorithms.Parameter {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public Parameter(DaalContext context, long cParameter) {
        super(context);
        this.cObject = cParameter;
    }

    /**
     * Sets the interceptFlag flag that enables or disables the computation
     * of the beta0 coefficient in the ridge regression equation
     * @param flag
     */
    public void setInterceptFlag(boolean flag) {
        cSetInterceptFlag(this.cObject, flag);
    }

    /**
     * Returns the value of the interceptFlag flag
     * @return Flag
     */
    public boolean getInterceptFlag() {
        return cGetInterceptFlag(this.cObject);
    }

    private native void cSetInterceptFlag(long algAddr, boolean flag);

    private native boolean cGetInterceptFlag(long algAddr);
}
/** @} */
