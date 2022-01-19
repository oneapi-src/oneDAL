/* file: BinaryConfusionMatrixMethod.java */
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
