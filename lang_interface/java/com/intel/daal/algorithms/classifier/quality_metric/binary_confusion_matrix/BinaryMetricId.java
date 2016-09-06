/* file: BinaryMetricId.java */
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
 * <a name="DAAL-CLASS-ALGORITHMS__CLASSIFIER__QUALITY_METRIC__BINARY_CONFUSION_MATRIX__BINARYMETRICID"></a>
 * @brief Available identifiers of binary metrics
 */
public final class BinaryMetricId {
    private int _value;

    public BinaryMetricId(int value) {
        _value = value;
    }

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
