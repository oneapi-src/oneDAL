/* file: QualityMetricId.java */
/*******************************************************************************
* Copyright 2014 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
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
