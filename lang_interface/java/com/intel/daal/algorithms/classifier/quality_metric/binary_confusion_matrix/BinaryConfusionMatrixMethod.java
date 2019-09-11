/* file: BinaryConfusionMatrixMethod.java */
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
 * @ingroup quality_metric_binary
 * @{
 */
package com.intel.daal.algorithms.classifier.quality_metric.binary_confusion_matrix;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__CLASSIFIER__QUALITY_METRIC__BINARY_CONFUSION_MATRIX__BINARYCONFUSIONMATRIXMETHOD"></a>
 * @brief Available methods for computing the binary confusion matrix
 */
public final class BinaryConfusionMatrixMethod {
    private int _value;

    /**
     * Constructs the binary confusion matrix method object using the provided value
     * @param value     Value corresponding to the binary confusion matrix method object
     */
    public BinaryConfusionMatrixMethod(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the binary confusion matrix method object
     * @return Value corresponding to the binary confusion matrix method object
     */
    public int getValue() {
        return _value;
    }

    private static final int DefaultDense = 0;

    public static final BinaryConfusionMatrixMethod defaultDense = new BinaryConfusionMatrixMethod(
            DefaultDense); /*!< Default method */
}
/** @} */
