/* file: MultiClassConfusionMatrixResultId.java */
/*******************************************************************************
* Copyright 2014-2022 Intel Corporation
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
 * @ingroup quality_metric_multiclass
 * @{
 */
package com.intel.daal.algorithms.classifier.quality_metric.multi_class_confusion_matrix;

import java.lang.annotation.Native;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__CLASSIFIER__QUALITY_METRIC__MULTI_CLASS_CONFUSION_MATRIX__MULTICLASSCONFUSIONMATRIXRESULTID"></a>
 * @brief Available identifiers of the results of the confusion matrix algorithm
 */
public final class MultiClassConfusionMatrixResultId {
    private int _value;

    /**
     * Constructs the confusion matrix result object identifier using the provided value
     * @param value     Value corresponding to the confusion matrix result object identifier
     */
    public MultiClassConfusionMatrixResultId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the confusion matrix result object identifier
     * @return Value corresponding to the confusion matrix result object identifier
     */
    public int getValue() {
        return _value;
    }

    @Native private static final int ConfusionMatrix   = 0;
    @Native private static final int MultiClassMetrics = 1;

    /*!< Labels computed in the prediction stage of the classification algorithm */
    public static final MultiClassConfusionMatrixResultId confusionMatrix   = new MultiClassConfusionMatrixResultId(
            ConfusionMatrix);
    /*!< Table that contains quality metrics (i.e., precision, recall, etc.) for multi-class classifiers */
    public static final MultiClassConfusionMatrixResultId multiClassMetrics = new MultiClassConfusionMatrixResultId(
            MultiClassMetrics);
}
/** @} */
