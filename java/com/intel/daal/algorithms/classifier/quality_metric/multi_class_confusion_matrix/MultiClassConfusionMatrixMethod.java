/* file: MultiClassConfusionMatrixMethod.java */
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
 * @ingroup quality_metric_multiclass
 * @{
 */
package com.intel.daal.algorithms.classifier.quality_metric.multi_class_confusion_matrix;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__CLASSIFIER__QUALITY_METRIC__MULTI_CLASS_CONFUSION_MATRIX__MULTICLASSCONFUSIONMATRIXMETHOD"></a>
 * @brief Available methods for computing the confusion matrix
 */
public final class MultiClassConfusionMatrixMethod {
    private int _value;

    /**
     * Constructs the confusion matrix method object using the provided value
     * @param value     Value corresponding to the confusion matrix method object
     */
    public MultiClassConfusionMatrixMethod(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the confusion matrix method object
     * @return Value corresponding to the confusion matrix method object
     */
    public int getValue() {
        return _value;
    }

    private static final int DefaultDense = 0;

    public static final MultiClassConfusionMatrixMethod defaultDense = new MultiClassConfusionMatrixMethod(
            DefaultDense); /*!< Default method */
}
/** @} */
