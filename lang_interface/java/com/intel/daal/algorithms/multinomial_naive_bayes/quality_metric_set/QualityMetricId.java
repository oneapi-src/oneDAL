/* file: QualityMetricId.java */
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
 * @ingroup multinomial_naive_bayes_quality_metric_set
 * @{
 */
package com.intel.daal.algorithms.multinomial_naive_bayes.quality_metric_set;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTINOMIAL_NAIVE_BAYES__QUALITY_METRIC_SET__QUALITYMETRICID"></a>
 * @brief Available identifiers of the quality metrics available for the model trained with the multinomial Naive Bayes algorithm
 */
public final class QualityMetricId {
    private int _value;

    /**
     * Constructs the quality metrics object identifier using the provided value
     * @param value     Value corresponding to the quality metrics object identifier
     */
    public QualityMetricId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the quality metrics object identifier
     * @return Value corresponding to the quality metrics object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int ConfusionMatrix = 0;

    public static final QualityMetricId confusionMatrix = new QualityMetricId(ConfusionMatrix); /*!< Confusion matrix */
}
/** @} */
