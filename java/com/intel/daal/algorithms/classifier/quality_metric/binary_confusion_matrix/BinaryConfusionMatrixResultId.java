/* file: BinaryConfusionMatrixResultId.java */
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
 * @ingroup quality_metric_binary
 * @{
 */
package com.intel.daal.algorithms.classifier.quality_metric.binary_confusion_matrix;

import java.lang.annotation.Native;

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

    @Native private static final int ConfusionMatrix = 0;
    @Native private static final int BinaryMetrics   = 1;

    /*!< Labels computed in the prediction stage of the classification algorithm */
    public static final BinaryConfusionMatrixResultId confusionMatrix = new BinaryConfusionMatrixResultId(
            ConfusionMatrix);
    /*!< Table that contains quality metrics (precision, recall, and so on) for binary classifiers */
    public static final BinaryConfusionMatrixResultId binaryMetrics   = new BinaryConfusionMatrixResultId(
            BinaryMetrics);
}
/** @} */
