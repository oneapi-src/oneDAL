/* file: TrainingMethod.java */
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
 * @defgroup logistic_regression_training Training
 * @brief Contains a class for logistic regression model-based training
 * @ingroup logistic_regression
 * @{
 */
package com.intel.daal.algorithms.logistic_regression.training;

import com.intel.daal.utils.*;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LOGISTIC_REGRESSION__TRAINING__TRAININGMETHOD"></a>
 * @brief Available methods for logistic regression model-based training
 */
public final class TrainingMethod {

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    private int _value;

    /**
     * Constructs the method object using the provided value
     * @param value     Value corresponding to the method object
     */
    public TrainingMethod(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the method object
     * @return Value corresponding to the method object
     */
    public int getValue() {
        return _value;
    }

    private static final int defaultDenseValue = 0;

    public static final TrainingMethod defaultDense = new TrainingMethod(
            defaultDenseValue);                                          /*!< Normal equations method */

}
/** @} */
