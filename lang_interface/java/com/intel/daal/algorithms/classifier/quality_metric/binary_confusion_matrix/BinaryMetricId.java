/* file: BinaryMetricId.java */
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
 * @ingroup quality_metric_binary
 * @{
 */
package com.intel.daal.algorithms.classifier.quality_metric.binary_confusion_matrix;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__CLASSIFIER__QUALITY_METRIC__BINARY_CONFUSION_MATRIX__BINARYMETRICID"></a>
 * @brief Available identifiers of binary metrics
 */
public final class BinaryMetricId {
    private int _value;

    /**
     * Constructs the binary metrics object identifier using the provided value
     * @param value     Value corresponding to the binary metrics object identifier
     */
    public BinaryMetricId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the binary metrics object identifier
     * @return Value corresponding to the binary metrics object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int Accuracy    = 0;
    private static final int Precision   = 1;
    private static final int Recall      = 2;
    private static final int Fscore      = 3;
    private static final int Specificity = 4;
    private static final int aUC         = 5;

    public static final BinaryMetricId accuracy    = new BinaryMetricId(Accuracy);
    public static final BinaryMetricId precision   = new BinaryMetricId(Precision);
    public static final BinaryMetricId recall      = new BinaryMetricId(Recall);
    public static final BinaryMetricId fscore      = new BinaryMetricId(Fscore);
    public static final BinaryMetricId specificity = new BinaryMetricId(Specificity);
    public static final BinaryMetricId AUC         = new BinaryMetricId(aUC);
}
/** @} */
