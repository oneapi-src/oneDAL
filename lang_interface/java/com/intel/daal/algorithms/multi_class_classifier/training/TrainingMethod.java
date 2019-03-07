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
 * @defgroup multi_class_classifier_training Training
 * @brief Contains classes for training the multi-class classifier model
 * @ingroup multi_class_classifier
 * @{
 */
package com.intel.daal.algorithms.multi_class_classifier.training;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTI_CLASS_CLASSIFIER__TRAINING__TRAININGMETHOD"></a>
 * @brief Computation methods for the multi-class classifier algorithm
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

    private static final int OneAgainstOne = 0;

    public static final TrainingMethod oneAgainstOne = new TrainingMethod(OneAgainstOne); /*!< One-Against-One method */
}
/** @} */
