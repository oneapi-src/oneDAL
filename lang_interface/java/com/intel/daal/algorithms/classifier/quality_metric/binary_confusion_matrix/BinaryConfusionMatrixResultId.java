/* file: BinaryConfusionMatrixResultId.java */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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

package com.intel.daal.algorithms.classifier.quality_metric.binary_confusion_matrix;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__CLASSIFIER__QUALITY_METRIC__BINARY_CONFUSION_MATRIX__BINARYCONFUSIONMATRIXRESULTID"></a>
 * @brief Available identifiers of results of the binary confusion matrix algorithm
 */
public final class BinaryConfusionMatrixResultId {
    private int _value;

    public BinaryConfusionMatrixResultId(int value) {
        _value = value;
    }

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
