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
 * @defgroup ridge_regression Ridge Regression
 * @brief Contains classes of the ridge regression algorithm
 * @ingroup regression
 * @{
 */
package com.intel.daal.algorithms.ridge_regression;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__RIDGE_REGRESSION__MODEL"></a>
 * @brief %Base class for models trained by the ridge regression training algorithm
 *
 * @par References
 *      - Parameter class
 *      - ModelNormEq class
 */
public class Model extends com.intel.daal.algorithms.Model {
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
     * Returns the number of features in the training data set
     * @return Number of features in the training data set
     */
    public long getNumberOfFeatures() {
        return cGetNumberOfFeatures(this.getCObject());
    }

    /**
     * Returns the number of responses in the training data set
     * @return Number of responses in the training data set
     */
    public long getNumberOfResponses() {
        return cGetNumberOfResponses(this.getCObject());
    }

    /**
     * Returns the numeric table that contains regression coefficients
     * @return Numeric table that contains regression coefficients
     */
    public NumericTable getBeta() {
        return (NumericTable)Factory.instance().createObject(getContext(), cGetBeta(this.cObject));
    }

    protected Precision prec;

    protected native long cGetBeta(long modelAddr);

    private native long cGetNumberOfFeatures(long modelAddr);

    private native long cGetNumberOfBetas(long modelAddr);

    private native long cGetNumberOfResponses(long modelAddr);
}
/** @} */
