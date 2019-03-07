/* file: BinaryConfusionMatrixResultId.java */
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
 * <a name="DAAL-CLASS-ALGORITHMS__CLASSIFIER__QUALITY_METRIC__BINARY_CONFUSION_MATRIX__BINARYCONFUSIONMATRIXRESULTID"></a>
 * @brief Available identifiers of results of the binary confusion matrix algorithm
 */
public final class BinaryConfusionMatrixResultId {
    private int _value;

    /**
     * Constructs the binary confusion matrix result object identifier using the provided value
     * @param value     Value corresponding to the binary confusion matrix result object identifier
     */
    public BinaryConfusionMatrixResultId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the binary confusion matrix result object identifier
     * @return Value corresponding to the binary confusion matrix result object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int ConfusionMatrix = 0;
    private static final int BinaryMetrics   = 1;

    /*!< Labels computed in the prediction stage of the classification algorithm */
    public static final BinaryConfusionMatrixResultId confusionMatrix = new BinaryConfusionMatrixResultId(
            ConfusionMatrix);
    /*!< Table that contains quality metrics (precision, recall, and so on) for binary classifiers */
    public static final BinaryConfusionMatrixResultId binaryMetrics   = new BinaryConfusionMatrixResultId(
            BinaryMetrics);
}
/** @} */
