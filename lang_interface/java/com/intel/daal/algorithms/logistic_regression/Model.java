/* file: Model.java */
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
 * @defgroup logistic_regression Logistic Regression
 * @brief Contains classes of the logistic regression algorithm
 * @ingroup regression
 * @{
 */
package com.intel.daal.algorithms.logistic_regression;

import com.intel.daal.utils.*;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LOGISTIC_REGRESSION__MODEL"></a>
 * @brief %Base class for models trained by the logistic regression training algorithm
 *
 * @par References
 *      - Parameter class
 */
public class Model extends com.intel.daal.algorithms.classifier.Model {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public Model(DaalContext context, long cModel) {
        super(context, cModel);
    }

    /**
     * Returns the number of regression coefficients
     * @return Number of regression coefficients
     */
    public long getNumberOfBetas() {
        return cGetNumberOfBetas(this.getCObject());
    }

    /**
     * Returns true if the regression model contains the intercept term, and false otherwise
     * @return True if the regression model contains the intercept term, and false otherwise
     */
    public boolean getInterceptFlag() {
        return cGetInterceptFlag(this.cObject);
    }

    /**
     * Sets the interceptFlag flag that enables or disables the computation
     * of the beta0 coefficient in the logistic regression equation
     * @param flag
     */
    public void setInterceptFlag(boolean flag) {
        cSetInterceptFlag(this.cObject, flag);
    }

    /**
     * Returns the numeric table that contains regression coefficients
     * @return Numeric table that contains regression coefficients
     */
    public NumericTable getBeta() {
        return (NumericTable)Factory.instance().createObject(getContext(), cGetBeta(this.cObject));
    }

    private native long cGetNumberOfBetas(long modelAddr);
    private native boolean cGetInterceptFlag(long modelAddr);
    private native void cSetInterceptFlag(long modelAddr, boolean flag);
    protected native long cGetBeta(long modelAddr);
}
/** @} */
