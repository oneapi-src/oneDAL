/* file: QualityMetricId.java */
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

package com.intel.daal.algorithms.linear_regression.quality_metric_set;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LINEAR_REGRESSION__QUALITY_METRIC_SET__QUALITYMETRICID"></a>
 * @brief Available identifiers of the quality metrics available for the model trained with the linear regression algorithm
 */
public final class QualityMetricId {
    private int _value;

    public QualityMetricId(int value) {
        _value = value;
    }

    public int getValue() {
        return _value;
    }

    private static final int SingleBeta = 0;
    private static final int GroupOfBetas = 1;

    public static final QualityMetricId singleBeta = new QualityMetricId(SingleBeta); /*!< Single coefficient metrics */
    public static final QualityMetricId groupOfBetas = new QualityMetricId(GroupOfBetas); /*!< Group of coefficients metrics */
}
