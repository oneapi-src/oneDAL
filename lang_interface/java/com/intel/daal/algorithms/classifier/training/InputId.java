/* file: InputId.java */
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
 * @defgroup training Training
 * @brief Contains classes for training the model of the classification algorithms
 * @ingroup classifier
 * @{
 */
/**
 * @brief Contains classes for training the classification model
 */
package com.intel.daal.algorithms.classifier.training;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__CLASSIFIER__TRAINING__INPUTID"></a>
 * @brief Available identifiers of input objects for the classifier algorithm
 */
public final class InputId {
    private int _value;

    /**
     * Constructs the input object identifier using the provided value
     * @param value     Value corresponding to the input object identifier
     */
    public InputId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the input object identifier
     * @return Value corresponding to the input object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int Data    = 0;
    private static final int Labels  = 1;
    private static final int Weights = 2;

    public static final InputId data    = new InputId(Data);    /*!< Data for the training stage */
    public static final InputId labels  = new InputId(Labels);  /*!< Labels for the training stage */
    public static final InputId weights = new InputId(Weights); /*!< Weights for the training stage */
}
/** @} */
