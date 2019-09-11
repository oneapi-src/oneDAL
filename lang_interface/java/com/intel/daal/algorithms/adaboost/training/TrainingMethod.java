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
 * @defgroup adaboost_training Training
 * @brief Contains classes for AdaBoost models training
 * @ingroup adaboost
 * @{
 */
package com.intel.daal.algorithms.adaboost.training;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__ADABOOST__TRAINING__TRAININGMETHOD"></a>
 * @brief Available methods for training AdaBoost models
 */
public final class TrainingMethod {

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

    private static final int DefaultDense = 0;

    /** Default AdaBoost training method */
    public static final TrainingMethod defaultDense = new TrainingMethod(DefaultDense);
}
/** @} */
