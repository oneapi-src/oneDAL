/* file: MultiClassMetricId.java */
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
 * @ingroup quality_metric_multiclass
 * @{
 */
package com.intel.daal.algorithms.classifier.quality_metric.multi_class_confusion_matrix;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__CLASSIFIER__QUALITY_METRIC__MULTI_CLASS_CONFUSION_MATRIX__MULTICLASSMETRICID"></a>
 * @brief Available identifiers of multi-class metrics
 */
public final class MultiClassMetricId {
    private int _value;

    /**
     * Constructs the multi-class metrics object identifier using the provided value
     * @param value     Value corresponding to the multi-class metrics object identifier
     */
    public MultiClassMetricId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the multi-class metrics object identifier
     * @return Value corresponding to the multi-class metrics object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int AverageAccuracy = 0;
    private static final int ErrorRate       = 1;
    private static final int MicroPrecision  = 2;
    private static final int MicroRecall     = 3;
    private static final int MicroFscore     = 4;
    private static final int MacroPrecision  = 5;
    private static final int MacroRecall     = 6;
    private static final int MacroFscore     = 7;

    public static final MultiClassMetricId averageAccuracy = new MultiClassMetricId(AverageAccuracy);
    public static final MultiClassMetricId errorRate       = new MultiClassMetricId(ErrorRate);
    public static final MultiClassMetricId microPrecision  = new MultiClassMetricId(MicroPrecision);
    public static final MultiClassMetricId microRecall     = new MultiClassMetricId(MicroRecall);
    public static final MultiClassMetricId microFscore     = new MultiClassMetricId(MicroFscore);
    public static final MultiClassMetricId macroPrecision  = new MultiClassMetricId(MacroPrecision);
    public static final MultiClassMetricId macroRecall     = new MultiClassMetricId(MacroRecall);
    public static final MultiClassMetricId macroFscore     = new MultiClassMetricId(MacroFscore);
}
/** @} */
