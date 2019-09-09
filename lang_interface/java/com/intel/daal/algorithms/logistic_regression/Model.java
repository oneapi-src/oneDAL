/* file: Model.java */
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
