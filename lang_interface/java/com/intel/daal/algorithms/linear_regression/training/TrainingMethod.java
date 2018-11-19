/* file: TrainingMethod.java */
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
 * @ingroup linear_regression_training
 * @{
 */
package com.intel.daal.algorithms.linear_regression.training;

import com.intel.daal.utils.*;
/**
 * <a name="DAAL-CLASS-ALGORITHMS__LINEAR_REGRESSION__TRAINING__TRAININGMETHOD"></a>
 * @brief Available methods for linear regression model-based training
 */
public final class TrainingMethod {

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    private int _value;

    /**
     * Constructs the training method object using the provided value
     * @param value     Value corresponding to the training method object
     */
    public TrainingMethod(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the training method object
     * @return Value corresponding to the training method object
     */
    public int getValue() {
        return _value;
    }

    private static final int defaultDenseValue = 0;
    private static final int normEqDenseValue  = 0;
    private static final int qrDenseValue      = 1;

    public static final TrainingMethod defaultDense = new TrainingMethod(
            defaultDenseValue);                                          /*!< Normal equations method */
    public static final TrainingMethod normEqDense  = new TrainingMethod(
            normEqDenseValue);                                           /*!< Normal equations method */
    public static final TrainingMethod qrDense      = new TrainingMethod(
            qrDenseValue);                                               /*!< QR decomposition-based method */
}
/** @} */
