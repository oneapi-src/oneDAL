/* file: MultiClassMetricId.java */
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

package com.intel.daal.algorithms.classifier.quality_metric.multi_class_confusion_matrix;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__CLASSIFIER__QUALITY_METRIC__MULTI_CLASS_CONFUSION_MATRIX__MULTICLASSMETRICID"></a>
 * @brief Available identifiers of multi-class metrics
 */
public final class MultiClassMetricId {
    private int _value;

    public MultiClassMetricId(int value) {
        _value = value;
    }

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
